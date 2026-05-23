# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict

import torch
from loguru import logger

import ttnn
from models.common.llama_models import (
    CompletionMessage,
    StopReason,
    TokenResult,
    create_vision_mask,
    encode_content,
    extract_images_from_messages,
    sample_top_p,
)
from models.common.model_capabilities import ModelCapabilitiesMixin
from models.common.sampling import (
    SamplingParams,
    broadcast_sampling_params,
    chunk_sampling_params,
    format_sampling_params,
)
from models.common.sampling.tt_log_probs import LogProbsResult, reformat_logprobs
from models.common.warmup import WarmupForwardMixin
from models.tt_transformers.tt.common import (
    Mode,
    copy_host_to_device,
    get_block_size,
    get_max_prefill_chunk_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)

# Maximum total tokens (batch_size * seq_len) allowed for a batched prefill pass.
# Exceeding this triggers a fallback to sequential per-user prefill.
MAX_BATCHED_PREFILL_SEQ_LEN = 128 * 1024

# Power-of-2 batch sizes supported by trace caching for batched prefill.
SUPPORTED_PREFILL_BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def max_prefill_chunk_size_cutoff(sequence_length, max_prefill_chunk_size):
    return sequence_length > max_prefill_chunk_size


def _deepseek_kvdbg_enabled() -> bool:
    return os.getenv("DEEPSEEK_KVDBG", "").lower() in ("1", "true", "yes", "y")


class Generator(ModelCapabilitiesMixin, WarmupForwardMixin):
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
        """
        Creating a LlamaVision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        LlamaVision is general to text and chat.

        For bringup, make this class general to any backend implementation, as long as it takes torch tensors and returns torch tensors.

        """
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.processor = processor
        self.tokenizer = tokenizer
        self.data_parallel = len(self.model)
        self.prev_page_table = None
        self.trace_id_prefill = defaultdict(lambda: None)
        self.trace_inputs_prefill = defaultdict(lambda: None)
        self.trace_output_prefill = defaultdict(lambda: None)
        self.trace_id_prefill_sampling = defaultdict(lambda: None)
        self.trace_input_prefill_sampling = defaultdict(lambda: None)
        self.trace_output_prefill_sampling = defaultdict(lambda: None)
        self.trace_ids_decode = defaultdict(lambda: None)  # {device_sampling_bool: {device_id: trace_id}}
        self.trace_inputs_decode = defaultdict(lambda: None)
        self.trace_output_decode = defaultdict(lambda: None)
        self.prefill_traces_warmup = False
        self.already_warmed_up_prefill = False
        # Tenstorrent-p1 (Vector 2): tracks whether switch_mode(Mode.DECODE) has been called.
        # When True, any new prefill trace capture runs under the DECODE sub-device manager,
        # so register_default_trace_on_active_manager() must NOT be called (the trace is already
        # in the DECODE manager's pool; calling remap would TT_FATAL on "trace not in DEFAULT").
        self._prefill_decode_mode_entered = False
        # Set of trace_keys (f"{prefill_seq_len}_{model_id}_{batch_size}") captured under the
        # DEFAULT manager — these need V2.5 suspend/resume on replay.
        self._prefill_trace_captured_under_default: set = set()
        # V2.5 (SubDevices.md §1.2): saved prefetcher state for suspend/resume around prefill.
        # Populated by _v25_suspend_prefetcher(); consumed (and cleared) by the restore block
        # in _decode_forward_trace_text at the start of the FIRST decode step after prefill.
        self._v25_saved_worker_sub_device_id: dict = {}   # model_id → value
        self._v25_saved_receiver_sub_device_id: dict = {}  # model_id → value
        self._v25_saved_prefetcher_mode: dict = {}         # model_id → Mode
        self._v25_decode_restore_pending: bool = False     # True = decode restore needed (legacy safety net)
        # V2.5 split-logits: set to model_id when DEFAULT manager is active after trace replay
        # and needs a mid-logits restore (slice under DEFAULT, then restore, then norm+lm_head).
        self._v25_split_logits_model_id: int = -1         # -1 = not active
        # By default, enable split sampling (break the decode trace into two parts: upto logits, then sampling step)
        self.enable_split_sampling = True
        self.mode = None

    # Class-level capabilities (VLLM specific, to be overridden by subclasses)
    model_capabilities = {
        "supports_prefix_caching": True,
    }

    def _set_sampling_trace_mode(self, enabled: bool):
        for model_instance in self.model:
            sampling_module = getattr(model_instance, "sampling", None)
            if sampling_module is not None:
                sampling_module.enable_internal_trace = enabled

    def _get_sampling_contract(self, model_id: int):
        sampling_module = getattr(self.model[model_id], "sampling", None)
        sampling_dp = getattr(self.model[model_id], "sampling_dp", 1)
        group_batch = sampling_module.tt_sampling.max_batch_size if sampling_module is not None else None
        total_sampling_batch = group_batch * sampling_dp if group_batch is not None else None
        return sampling_module, sampling_dp, group_batch, total_sampling_batch

    def _mock_tokens(self, batch_size, seq_len, kv_cache, model_id):
        ret = dict()
        ret["tokens"] = torch.zeros(batch_size, seq_len, dtype=torch.long)
        ret["prompt_lens"] = torch.tensor([seq_len] * batch_size, dtype=torch.long)
        ret["empty_slots"] = list(range(batch_size))

        page_table_warmup = None
        # second check is some tests set the kv_cache to [None] instead of None
        if kv_cache is not None and kv_cache[model_id] is not None:
            block_size = get_block_size(kv_cache[model_id])
            num_blocks = num_blocks_in_seq(seq_len, block_size)
            page_table_warmup = torch.zeros(batch_size, num_blocks, dtype=torch.int32)

        ret["page_table"] = page_table_warmup

        return ret

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device):
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        # Tenstorrent-p1: Warmup runs before switch_mode(DECODE), so the prefill trace is
        # captured under DEFAULT manager.  Warmup must capture the trace AND run
        # process_logits_after_prefill_trace so that post-prefill ops (ttnn.slice, lm_head)
        # are compiled under DEFAULT manager.  These DEFAULT-compiled entries are then reused
        # when post-prefill ops run under DECODE manager on subsequent requests.
        # RESULT: warmup trace capture is kept (enable_trace unchanged).

        sequence_lengths_to_warmup = self.model_args[0].get_warmup_prefill_supported_seq_lens()
        warmup_batch_sizes = (1,)

        skip_sequence_lengths = False

        # Sweep all sampling parameters for prefill warmup just once since it is sequence length agnostic
        sampling_parameters_sweeped = False

        if enable_trace:
            logger.info("Using batch-1-only traced prefill warmup; runtime batched prefill remains enabled")

        for model_id in range(self.data_parallel):
            for supported_length in sequence_lengths_to_warmup:
                if model_id != 0 and (
                    supported_length not in self.model_args[0].trace_prefill_supported_seq_lens or not enable_trace
                ):
                    continue

                # Token-limit guard below skips combinations that would
                # exceed MAX_BATCHED_PREFILL_SEQ_LEN.
                for batch_size in warmup_batch_sizes:
                    if batch_size > 1 and batch_size * supported_length >= MAX_BATCHED_PREFILL_SEQ_LEN:
                        logger.info(
                            f"Skipping batched prefill warmup for batch_size={batch_size}, "
                            f"seq_len={supported_length}: exceeds token limit"
                        )
                        continue

                    warmup_args = self._mock_tokens(batch_size, supported_length, kv_cache, model_id)

                    # chunked prefill not supported without paged attention
                    if warmup_args["page_table"] is None and max_prefill_chunk_size_cutoff(
                        supported_length, self.model_args[0].max_prefill_chunk_size
                    ):
                        logger.warning(
                            f"Skipping warmup for sequence lengths after: {supported_length} because they are greater than the max prefill chunk size and paged attention is disabled"
                        )
                        skip_sequence_lengths = True
                        break

                    if not sampling_parameters_sweeped:
                        sampling_params = self._create_sampling_params(
                            can_sample_on_device=can_sample_on_device,
                            non_greedy_decoding_on_device=non_greedy_decoding_on_device,
                            batch_size=batch_size,
                        )
                    else:
                        sampling_params = [None]

                    for param in sampling_params:
                        logger.info(
                            f"Warming up prefill for sequence length: {supported_length} for batch size: {batch_size} with sampling params: {param}"
                        )
                        self.prefill_forward_text(
                            **warmup_args,
                            kv_cache=kv_cache,
                            enable_trace=enable_trace,
                            model_id_warmup=model_id,
                            sampling_params=param,
                        )

                    sampling_parameters_sweeped = True

                if skip_sequence_lengths:
                    break

        # Vision compile for multimodal models
        if getattr(self.model_args[0], "is_multimodal", False):
            vision_chunk_size = getattr(self.model_args[0], "vision_chunk_size", 896)
            vision_channels = getattr(self.model_args[0], "vision_in_channels", 3)
            model_id = 0

            # Create synthetic image for vision warmup
            # pixel_values is a list (one per user), each element is (num_images, C, H, W)
            warmup_pixel_values = [torch.zeros((1, vision_channels, vision_chunk_size, vision_chunk_size))]

            # Minimal text tokens for vision warmup pass, prefill expects non-empty tokens
            batch_size = 1  # VLMs support only batch=1 for now
            prefill_forward_args = self._mock_tokens(batch_size, 128, kv_cache, model_id)

            logger.info(f"Warming up vision encoder with image size {vision_chunk_size}x{vision_chunk_size}")

            self.prefill_forward_text(
                **prefill_forward_args,
                kv_cache=kv_cache,
                enable_trace=False,  # Vision encoder warmup doesn't support trace
                model_id_warmup=model_id,
                sampling_params=None,
                pixel_values=warmup_pixel_values,
                image_sizes=[(vision_chunk_size, vision_chunk_size)],
            )
            logger.info("Vision encoder warmup completed")

    def _capture_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        kv_cache=None,
        model_id=-1,
        global_user_id=None,
        batch_size=1,
        user_id=0,
    ):
        if batch_size > 1:
            prefill_kwargs = {"page_table": page_table, "batch_size": batch_size, "user_id": user_id}
            if global_user_id is not None:
                prefill_kwargs["global_user_id"] = global_user_id
            host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
            # These matrices will actually be pointing to the whole cos_matrix and sin_matrix that was allocated on device in the RotarySetup class
            tt_rot_mats_prefill_global = host_inputs[1]
            tt_rot_mats_prefill_local = host_inputs[2]
            host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
            transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=transformed_inputs[0],
                rot_mats_global=tt_rot_mats_prefill_global,
                rot_mats_local=tt_rot_mats_prefill_local,
                page_table=transformed_inputs[1],
                chunk_page_table=transformed_inputs[2],
                kv_cache=kv_cache,
                batch_size=batch_size,
                user_id=user_id,
            )
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)
            logger.info("Done Compiling Model")

            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
            trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
            transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=transformed_inputs[0],
                rot_mats_global=tt_rot_mats_prefill_global,
                rot_mats_local=tt_rot_mats_prefill_local,
                page_table=transformed_inputs[1],
                chunk_page_table=transformed_inputs[2],
                kv_cache=kv_cache,
                batch_size=batch_size,
                user_id=user_id,
            )
            ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)
            logger.info("Done Capturing Prefill Trace")
            return trace_id, tt_out_trace, *device_inputs
        else:
            prefill_kwargs = {"page_table": page_table}
            if global_user_id is not None:
                prefill_kwargs["global_user_id"] = global_user_id
            host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
            tt_rot_mats_prefill_global = host_inputs[1]
            tt_rot_mats_prefill_local = host_inputs[2]
            host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
            transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=transformed_inputs[0],
                rot_mats_global=tt_rot_mats_prefill_global,
                rot_mats_local=tt_rot_mats_prefill_local,
                page_table=transformed_inputs[1],
                chunk_page_table=transformed_inputs[2],
                kv_cache=kv_cache,
            )
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)
            logger.info("Done Compiling Model")

            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
            trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
            transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=transformed_inputs[0],
                rot_mats_global=tt_rot_mats_prefill_global,
                rot_mats_local=tt_rot_mats_prefill_local,
                page_table=transformed_inputs[1],
                chunk_page_table=transformed_inputs[2],
                kv_cache=kv_cache,
            )
            ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)
            logger.info("Done Capturing Prefill Trace")
            return trace_id, tt_out_trace, *device_inputs

    def _capture_trace_prefill_sampling(self, model_id, sampling_batch):
        """Capture a trace for batched prefill post-processing: norm + lm_head + sampling.

        Input buffer: [1, 1, sampling_batch, full_dim] host → column-sharded to
        [1, 1, sampling_batch, dim_per_device].
        Output: (tt_tokens, tt_log_probs) from sampling.
        """
        mesh_device = self.model_args[model_id].mesh_device
        full_dim = self.model_args[model_id].dim

        dummy_input = ttnn.from_torch(
            torch.zeros(1, 1, sampling_batch, full_dim, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )

        logits = self.model[model_id]._apply_norm_and_lm_head(dummy_input)
        tt_tokens, tt_log_probs = self.model[model_id].sampling.sample(logits, enable_trace=False)
        ttnn.synchronize_device(mesh_device)
        logger.info("Done compiling prefill sampling")

        trace_input = ttnn.from_torch(
            torch.zeros(1, 1, sampling_batch, full_dim, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        logits = self.model[model_id]._apply_norm_and_lm_head(trace_input)
        tt_tokens, tt_log_probs = self.model[model_id].sampling.sample(logits, enable_trace=False)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        logger.info("Done capturing prefill sampling trace")

        return trace_id, (tt_tokens, tt_log_probs), trace_input

    def _row_sharded_batched_prefill(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        prefill_seq_lens,
        enable_trace=True,
        sampling_params=None,
    ):
        """Dispatch to model's row-sharded batched prefill."""
        assert (
            self.data_parallel == 1
        ), "Row-sharded batched prefill requires data_parallel=1 (model handles DP internally)"
        return self.model[0].row_sharded_batched_prefill(
            tokens,
            page_table,
            kv_cache[0],
            prompt_lens,
            prefill_seq_lens,
            enable_trace=enable_trace,
            sampling_params=sampling_params,
            model_args=self.model_args[0],
            trace_cache={
                "ids": self.trace_id_prefill,
                "inputs": self.trace_inputs_prefill,
                "outputs": self.trace_output_prefill,
            },
        )

    def _easy_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        user_id=0,
        last_token_idx=None,
        kv_cache=None,
        model_id=-1,
        prefill_seq_len=None,
        batch_size=1,
        **kwargs,
    ):
        global_user_id = kwargs.get("global_user_id", None)
        trace_key = f"{prefill_seq_len}_{model_id}_{batch_size}"
        if self.trace_id_prefill[trace_key] is None:
            # Tenstorrent-p1 (V2.5): Prefill trace capture.
            # Warmup traces are captured under DEFAULT manager (before DECODE mode is set).
            # After DECODE mode is entered (DECODE manager active + prefetcher running),
            # we suspend the prefetcher (V2.5) and capture under DEFAULT manager as well,
            # so ALL prefill traces live in the DEFAULT pool (no GlobalCB L1 clash).
            # V2.5 replay switches to DEFAULT manager → no cross-manager remap needed.
            if self._prefill_decode_mode_entered:
                # DECODE manager is active; suspend to DEFAULT before capture.
                # (Restore happens in _prefill_forward_trace after execute_trace.)
                self._v25_suspend_prefetcher(model_id)

            trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
                prefill_ids,
                page_table=page_table,
                kv_cache=kv_cache,
                model_id=model_id,
                global_user_id=global_user_id,
                batch_size=batch_size,
                user_id=user_id,
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out_trace
            # ALL prefill traces are captured under DEFAULT manager (pre or post DECODE mode).
            self._prefill_trace_captured_under_default.add(trace_key)
            if not self._prefill_decode_mode_entered:
                logger.info(f"[V2.5] Prefill trace key={trace_key} captured under DEFAULT manager (warmup, pre-decode)")
            else:
                logger.info(f"[V2.5] Prefill trace key={trace_key} captured under DEFAULT manager (post-decode-mode, V2.5 suspended)")

        tt_out_trace = self._prefill_forward_trace(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            prefill_ids,
            page_table=page_table,
            model_id=model_id,
            global_user_id=global_user_id,
            batch_size=batch_size,
            user_id=user_id,
            trace_key=trace_key,
        )

        return tt_out_trace

    def _prefill_forward_trace(
        self,
        trace_id,
        device_inputs,
        tt_out_trace,
        prefill_ids,
        user_id=0,
        page_table=None,
        model_id=-1,
        global_user_id=None,
        batch_size=1,
        trace_key=None,
    ):
        # Use actual batch_size since tokens are now in batch dimension
        prefill_kwargs = {"page_table": page_table, "batch_size": batch_size, "user_id": user_id}
        if global_user_id is not None:
            prefill_kwargs["global_user_id"] = global_user_id
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

        device_inputs = copy_host_to_device(
            host_inputs, device_tensors=device_inputs, mesh_device=self.model_args[model_id].mesh_device
        )

        # Tenstorrent-p1 (V2.5): prefill trace replay.
        #
        # V2.5 (SubDevices.md §1.2): for prefill traces captured under DEFAULT manager
        # (warmup or post-DECODE-mode), suspend the prefetcher (switch to DEFAULT manager)
        # before execute_trace, then defer restore to the next decode step.
        #
        # This avoids BOTH problems that killed Options A and B:
        #   - Stream-0 per-device counter mismatch (cross-manager remap hangs on multi-chip)
        #   - GlobalCB L1 clash (no GlobalCB exists under DEFAULT manager)
        #
        # For pre-DECODE warmup replays, _v25_decode_restore_pending stays False (no restore
        # needed since DEFAULT manager is already active and prefetcher not yet running).
        mesh_device = self.model_args[model_id].mesh_device
        _is_captured_under_default = trace_key in self._prefill_trace_captured_under_default if trace_key is not None else False
        if _is_captured_under_default and self._prefill_decode_mode_entered:
            # Post-DECODE prefill replay: DECODE manager is active, prefetcher running.
            #
            # V2.5 strategy (full-DEFAULT path, Layer 20 v2):
            #   1. Suspend prefetcher (switch to DEFAULT manager) — avoids sub-device
            #      intersection errors during trace replay (trace was captured under DEFAULT).
            #   2. Execute prefill trace non-blocking under DEFAULT manager.
            #   3. Set _v25_split_logits_model_id so prefill_forward_text runs all
            #      post-trace logit processing under DEFAULT:
            #        a. ttnn.slice under DEFAULT (full device grid required for slice kernels)
            #        b. _apply_norm_and_lm_head under DEFAULT (no GlobalCB clash because
            #           Layer 20 v2 removes global-allocator check from
            #           lowest_occupied_compute_l1_address — receiver cores and compute
            #           cores have independent L1 address spaces, so no real clash exists)
            #        c. _v25_restore_prefetcher restores DECODE manager AFTER lm_head
            #
            # Why blocking=False?
            #   Under DEFAULT manager, blocking=True calls finish_nolock() with stall-group
            #   = all-cores, which includes persistent sender/receiver kernels → hang.
            #   Non-blocking returns immediately; the subsequent ttnn.slice and lm_head
            #   ops will naturally wait for the trace output tensor to be ready.
            logger.info(f"[V2.5] Prefill trace (key={trace_key}): suspending prefetcher for replay under DEFAULT manager")
            self._v25_suspend_prefetcher(model_id)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            # Do NOT restore DECODE manager here.  Both ttnn.slice and lm_head will run
            # under DEFAULT manager in prefill_forward_text (split-logits path), and DECODE
            # manager is restored only after lm_head completes.
            self._v25_split_logits_model_id = model_id
            logger.info(f"[V2.5] Prefill trace (key={trace_key}) replayed under DEFAULT; full-DEFAULT split pending (model_id={model_id})")
        else:
            # Pre-DECODE warmup replay (DEFAULT manager already active) or post-DECODE native capture.
            logger.info(f"[V2.5] Prefill trace (key={trace_key}): replaying directly (pre-decode or native capture)")
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

        return tt_out_trace

    def _v25_suspend_prefetcher(self, model_id: int):
        """V2.5: suspend prefetcher state and switch to DEFAULT manager for prefill replay.

        Saves worker_sub_device_id, receiver_sub_device_id, and prefetcher.mode, then
        zeros them out (so model.py skips prefetcher-specific paths) and calls
        prefetcher.load_default_manager() to clear the DECODE sub-device manager.

        No-op if model has no prefetcher or prefetcher is already suspended.
        """
        prefetcher = getattr(self.model[model_id], "prefetcher", None)
        if prefetcher is None:
            return
        # Only save once (idempotent)
        if model_id in self._v25_saved_prefetcher_mode:
            logger.info(f"[V2.5] suspend: model_id={model_id} already suspended, skipping")
            return
        self._v25_saved_worker_sub_device_id[model_id] = getattr(prefetcher, "worker_sub_device_id", None)
        self._v25_saved_receiver_sub_device_id[model_id] = getattr(prefetcher, "receiver_sub_device_id", None)
        self._v25_saved_prefetcher_mode[model_id] = getattr(prefetcher, "mode", None)
        # Zero out so model.py skip prefetcher paths during prefill replay
        prefetcher.worker_sub_device_id = None
        prefetcher.receiver_sub_device_id = None
        prefetcher.mode = Mode.PREFILL
        # V2.5 [Layer 21]: Drain ALL sub-devices (sender=0, receiver=1, worker=2) BEFORE
        # switching to DEFAULT manager.
        #
        # Root cause of v12 hang: after the decode trace finishes with
        # finish_nolock([stall_group={sub_device_2}]), the WORKER completion acks
        # (stream 2) have arrived at M_delta, but the SENDER (stream 0) and RECEIVER
        # (stream 1) kernel acks may still be in flight.  When
        # reset_worker_dispatch_state_on_device is called inside
        # clear_loaded_sub_device_manager(), it sends:
        #   add_dispatch_go_signal_mcast(expected[0], RESET_READ_PTR, stream_0, ...)
        # The dispatch firmware waits for stream 0 == expected[0] before multicasting.
        # If stream 0 < expected[0] (sender still running), it hangs forever.
        #
        # Fix: synchronize ALL 3 sub-devices here, so streams 0, 1, 2 all reach
        # their expected counts before we switch to DEFAULT.  The synchronization
        # enqueues WAIT_STREAM for each of the 3 streams and blocks until all fire.
        all_sub_device_ids = prefetcher.prefetcher_sub_device.sub_devices_id
        mesh_device = self.model_args[model_id].mesh_device
        logger.info(
            f"[V2.5] suspend: draining all DECODE sub-devices {all_sub_device_ids} "
            f"before DEFAULT switch (model_id={model_id})"
        )
        import ttnn as _ttnn_v25
        _ttnn_v25.synchronize_device(mesh_device, sub_device_ids=all_sub_device_ids)
        logger.info(f"[V2.5] suspend: all DECODE sub-devices drained, switching to DEFAULT manager")
        # Switch to DEFAULT manager (clears DECODE sub-device manager + GlobalCB not active)
        prefetcher.load_default_manager()
        logger.info(
            f"[V2.5] suspend: model_id={model_id} saved worker={self._v25_saved_worker_sub_device_id[model_id]} "
            f"receiver={self._v25_saved_receiver_sub_device_id[model_id]} mode={self._v25_saved_prefetcher_mode[model_id]}"
        )

    def _v25_restore_prefetcher(self, model_id: int):
        """V2.5: restore prefetcher state and switch back to DECODE manager.

        Called at the start of the first decode step after a V2.5-suspended prefill.
        Restores worker_sub_device_id, receiver_sub_device_id, prefetcher.mode, and
        calls prefetcher.load_decode_manager() to re-activate the DECODE sub-device manager.

        No-op if model has no prefetcher or nothing was saved.
        """
        prefetcher = getattr(self.model[model_id], "prefetcher", None)
        if prefetcher is None:
            return
        if model_id not in self._v25_saved_prefetcher_mode:
            logger.info(f"[V2.5] restore: model_id={model_id} nothing saved, skipping")
            return
        prefetcher.worker_sub_device_id = self._v25_saved_worker_sub_device_id.pop(model_id)
        prefetcher.receiver_sub_device_id = self._v25_saved_receiver_sub_device_id.pop(model_id)
        prefetcher.mode = self._v25_saved_prefetcher_mode.pop(model_id)
        # Restore DECODE sub-device manager (re-restricts stall group to worker-only)
        prefetcher.load_decode_manager()
        logger.info(
            f"[V2.5] restore: model_id={model_id} restored worker={prefetcher.worker_sub_device_id} "
            f"receiver={prefetcher.receiver_sub_device_id} mode={prefetcher.mode}"
        )

    # Note: This function is called by vLLM
    def prefill_forward_text(
        self,
        tokens: torch.Tensor,  # All tokens, including the cached ones
        page_table=None,
        kv_cache=None,
        prompt_lens=None,  # Full prompt lengths, including the cached ones
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params: SamplingParams | None = None,
        start_pos: list[int] = None,  # Cached prefixes lengths
        return_hidden_states=False,
        warmup_prefill=True,
        **kwargs,
    ):
        self.mode = Mode.PREFILL
        # SGLANG_TT_DISABLE_PREFILL_TRACE=1: skip prefill trace capture/replay entirely.
        # Prefill runs untraced (3× slower TTFT) but avoids cross-manager trace replay
        # that hangs run2 when the prefetcher is active.  Decode traces still work,
        # so TPOT (the user-facing metric) retains the prefetcher speedup.
        if os.environ.get("SGLANG_TT_DISABLE_PREFILL_TRACE", "0") != "0":
            enable_trace = False
            logger.info("[PREFILL-TRACE] SGLANG_TT_DISABLE_PREFILL_TRACE=1: prefill trace disabled, using untraced forward")
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        else:
            # Only paged attention is supported for prefill
            enable_trace = False

        sampling_on_device_requested = sampling_params is not None

        # we need this here because of tt-metal tests
        # WS-A.18: SGLANG_TT_DISABLE_PREFILL_WARMUP=1 skips the internal
        # warmup loop (which probes seq_lens up to context_length using
        # _mock_tokens). On Qwen3.5-0.8B the DeltaNet host fallback at
        # seq_len=2048 OOMs the process; bypassing the warmup lets the
        # first real request just compile what it needs. Identity for
        # every other model (Qwen3-8B etc.) when the env var is unset.
        if warmup_prefill and os.environ.get("SGLANG_TT_DISABLE_PREFILL_WARMUP", "0") != "1":
            sampling_on_device_enabled = (
                getattr(self.model[0], "_supports_on_device_sampling", False)
                and getattr(self.model[0], "sampling", None) is not None
            )

            self.warmup_model_prefill(
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                can_sample_on_device=sampling_on_device_enabled,
                non_greedy_decoding_on_device=sampling_on_device_enabled,
            )

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size

        # Output shape depends on whether we're returning logits or hidden states
        if return_hidden_states:
            # For hidden states, output shape is [batch_size, hidden_size]
            # Note: dim is the hidden dimension size
            hidden_size = self.model_args[0].dim
            output_tensor = torch.zeros(batch_size, hidden_size)
        else:
            # Each model expected to run the same model, safe to use 1st vocab size
            output_tensor = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)
            output_tokens = torch.zeros(batch_size, 1, dtype=torch.int64)
            output_log_probs = [None] * batch_size
        sampling_executed = False
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        # For row-sharded users, use max_local_batch_size (users per row) for group_user_id
        local_batch_size = getattr(self.model_args[0], "max_local_batch_size", max_batch_size_per_model)

        if not isinstance(prompt_lens, list):
            prompt_lens = prompt_lens.tolist()

        prefill_seq_lens = [get_padded_prefill_len(seq_len) for seq_len in prompt_lens]
        # Row-sharded batched prefill: process 1 user per row per iteration.
        # Only used when device sampling is active (sampling_params is not None)
        # and the prompt uses the harmony chat template (first token is <|start|>=200006).
        # Host sampling (sampling_params=None) needs the single-user prefill path
        # that returns full logits per user.
        model_0 = self.model[0]
        is_harmony = tokens.shape[1] > 0 and int(tokens[0, 0]) == 200006
        if (
            getattr(model_0, "users_row_sharded", False)
            and batch_size > 1
            and sampling_params is not None
            and is_harmony
        ):
            return self._row_sharded_batched_prefill(
                tokens,
                page_table,
                kv_cache,
                prompt_lens,
                prefill_seq_lens=prefill_seq_lens,
                enable_trace=enable_trace,
                sampling_params=sampling_params,
            )

        # Batched prefill: all prompts share the same padded length so they can
        # be processed in a single forward pass. padded_batch is rounded up to
        # the nearest SUPPORTED_PREFILL_BATCH_SIZES entry (not max_batch_size)
        # to keep all_gather buffers within DRAM limits.
        use_batched_prefill = (
            batch_size > 1
            and len(set(prefill_seq_lens)) == 1
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
        )

        if use_batched_prefill and sampling_on_device_requested:
            sampling_module, sampling_dp, _, _ = self._get_sampling_contract(0)
            if sampling_module is not None and sampling_dp > 1:
                # NOTE: Batched prefill disabled: on-device sampling
                # must fall back to sequential prefill until a row-sharded
                # batched-prefill sampling contract is implemented.
                use_batched_prefill = False

        if use_batched_prefill:
            padded_batch = next(
                (b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size),
                self.model_args[0].max_batch_size,
            )
            if padded_batch > self.model_args[0].max_batch_size:
                logger.info(
                    f"Batched prefill disabled: padded_batch {padded_batch} exceeds "
                    f"max_batch_size {self.model_args[0].max_batch_size}"
                )
                use_batched_prefill = False
            elif padded_batch * prefill_seq_lens[0] >= MAX_BATCHED_PREFILL_SEQ_LEN:
                logger.info(
                    f"Batched prefill disabled: {padded_batch} x {prefill_seq_lens[0]} = "
                    f"{padded_batch * prefill_seq_lens[0]} tokens exceeds limit {MAX_BATCHED_PREFILL_SEQ_LEN}"
                )
                use_batched_prefill = False

        if not use_batched_prefill:
            padded_batch = self.model_args[0].max_batch_size

        all_users = [0] if use_batched_prefill else empty_slots

        sampling_params_per_out: list[SamplingParams | None] = [None] * len(empty_slots)
        prompt_tokens_per_out: list[torch.Tensor | None] = [None] * len(empty_slots)
        prefill_results: list[dict] = []

        for idx, user_id in enumerate(all_users):
            model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup
            group_user_id = user_id % local_batch_size if page_table is None else 0

            if use_batched_prefill:
                batch_user_ids = empty_slots
                last_token_idx = [(seq_len - 1) for seq_len in prompt_lens]
                prefill_seq_len = prefill_seq_lens[0]
                seq_len = prompt_lens
            else:
                batch_user_ids = None
                seq_len = int(prompt_lens[idx])
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                last_token_idx = seq_len - 1
                prefill_seq_len = prefill_seq_lens[idx]
                logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")
            local_kwargs = kwargs.copy()  # Avoid modifying original kwargs
            if getattr(self.model[model_id], "users_row_sharded", False):
                local_kwargs["global_user_id"] = batch_user_ids if use_batched_prefill else user_id
            sampling_enabled = (
                sampling_on_device_requested
                and getattr(self.model[model_id], "_supports_on_device_sampling", False)
                and getattr(self.model[model_id], "sampling", None) is not None
            )

            if use_batched_prefill:
                # Galaxy 70B approach: slot-based placement with shape [padded_batch, prefill_seq_len]
                # Each request is placed at its corresponding slot index
                prefill_ids = torch.zeros(padded_batch, prefill_seq_len, dtype=torch.long, device=tokens.device)
                padded_last_token_idx = [0] * padded_batch  # dummy idx for padded slots
                for local_idx, slot in enumerate(empty_slots):
                    seq_len_local = int(seq_len[local_idx])
                    padded_tokens = torch.cat(
                        [
                            tokens[local_idx : local_idx + 1, :seq_len_local],
                            torch.zeros(1, prefill_seq_len - seq_len_local, dtype=torch.long, device=tokens.device),
                        ],
                        dim=-1,
                    )
                    prefill_ids[slot : slot + 1] = padded_tokens
                    padded_last_token_idx[slot] = last_token_idx[local_idx]
                last_token_idx = padded_last_token_idx
            else:
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                prefill_ids = torch.cat(
                    [
                        tokens[idx : idx + 1, num_cached_tokens:seq_len],
                        torch.zeros(1, prefill_seq_len - (seq_len - num_cached_tokens)).long(),
                    ],
                    dim=-1,
                )

            enable_trace_current_prompt = enable_trace and self.model_args[model_id].can_enable_trace(
                prefill_seq_len, num_cached_tokens if not use_batched_prefill else 0
            )

            # Tenstorrent-p1 (Layer 20): always use prefill trace when available.
            # Trace replay doesn't compile new programs → no static-CB clash with GlobalCB.
            # Warmup-captured traces (DEFAULT manager) are remapped via
            # register_default_trace_on_active_manager() at replay time when DECODE manager is active.
            logger.info(
                f"Prefill seq len: {prefill_seq_len}, max_prefill_chunk_size: {self.model_args[0].max_prefill_chunk_size}, trace: {enable_trace_current_prompt}"
            )

            if page_table is not None:
                # For batched prefill: pass full page_table (function handles slot placement)
                # For non-batched prefill: pass sliced page_table for current user (like original code)
                page_table_for_user = page_table if use_batched_prefill else page_table[idx : idx + 1]
                page_table_user = self._get_prefill_user_page_table(
                    page_table_for_user,
                    kv_cache[model_id],
                    seq_len,
                    trace_enabled=enable_trace_current_prompt,
                    prefill_seq_len=prefill_seq_len,
                    use_batched_prefill=use_batched_prefill,
                    user_id=batch_user_ids if use_batched_prefill else user_id,
                    padded_batch_size=padded_batch if use_batched_prefill else None,
                )
            else:
                page_table_user = None
            if page_table_user is not None and _deepseek_kvdbg_enabled():
                sample = []
                if page_table_user.numel():
                    flat = page_table_user.reshape(-1)
                    sample = flat[: min(16, flat.numel())].tolist()
                logger.debug(
                    "KVDBG deepseek prefill user global={} local={} seq_len={} cached={} page_table_shape={} sample={}",
                    user_id,
                    group_user_id,
                    seq_len,
                    num_cached_tokens,
                    list(page_table_user.shape),
                    sample,
                )
            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None

            # Check if 'pixel_values' exists and index it safely
            if local_kwargs.get("pixel_values", None) is not None:
                local_kwargs["pixel_values"] = local_kwargs["pixel_values"][idx]
                if "image_grid_thw" in local_kwargs:
                    local_kwargs["image_grid_thw"] = local_kwargs["image_grid_thw"][idx]
                if "image_sizes" in local_kwargs and local_kwargs["image_sizes"] is not None:
                    local_kwargs["image_sizes"] = local_kwargs["image_sizes"][idx]

            if sampling_enabled and not use_batched_prefill:
                sampling_executed = True
                sampling_dp = getattr(self.model[model_id], "sampling_dp", 1)
                total_batch = self.model[model_id].sampling.tt_sampling.max_batch_size * sampling_dp
                per_request_params = format_sampling_params(
                    broadcast_sampling_params(sampling_params, idx, slot_len=total_batch), total_batch
                )
                assert per_request_params is not None, "Sampling was executed but missing per-request sampling params"
                # empty_slots uses max_batch_size_per_model (not total_batch) because
                # the seed manager operates on per-row slots (0..31).  When sampling_dp > 1
                # the params are already broadcast across all rows by broadcast_sampling_params.
                self.model[model_id].sampling.apply_prefill_state(
                    sampling_params=per_request_params,
                    prompt_tokens=prefill_ids[:, :seq_len].repeat(total_batch, 1),
                    empty_slots=[user_id % max_batch_size_per_model],
                )

            if enable_trace_current_prompt:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=batch_user_ids if use_batched_prefill else group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    prefill_seq_len=prefill_seq_len,
                    batch_size=padded_batch if use_batched_prefill else 1,
                    **local_kwargs,
                )
            else:
                # Untraced prefill path. When the prefetcher is running under the
                # DECODE sub-device manager, untraced program compilation/execution
                # AND the subsequent ttnn.untilize both fail because the DECODE
                # manager restricts which cores are visible
                # ("Kernel group cores do not match sub device cores").
                #
                # Strategy: suspend the prefetcher (switch to DEFAULT manager) before
                # the untraced forward AND leave DEFAULT active through ttnn.untilize
                # and logits.cpu(). Use the V2.5 deferred-restore mechanism
                # (_v25_split_logits_model_id) so that DECODE is restored at line 1131
                # (after untilize + blocking cpu), exactly as in the traced path.
                _needs_manager_switch = self._prefill_decode_mode_entered and getattr(
                    self.model[model_id], "prefetcher", None
                ) is not None
                if _needs_manager_switch:
                    logger.info(
                        f"[PREFILL-TRACE-UNTRACED] DECODE manager active — suspending prefetcher "
                        f"for untraced prefill (model_id={model_id}); DECODE restore deferred to post-untilize"
                    )
                    self._v25_suspend_prefetcher(model_id)
                    # Mark for deferred restore (same slot as traced path uses).
                    # The restore at line ~1131 will call _v25_restore_prefetcher
                    # after untilize+cpu complete under DEFAULT manager.
                    self._v25_split_logits_model_id = model_id
                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=batch_user_ids if use_batched_prefill else group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    num_cached_tokens=0 if use_batched_prefill else num_cached_tokens,
                    batch_size=padded_batch if use_batched_prefill else 1,
                    **local_kwargs,
                )
            if use_batched_prefill:
                hidden_dim = logits.shape[-1]
                logits = ttnn.reshape(logits, [padded_batch, 1, prefill_seq_len, hidden_dim])

                if sampling_enabled:
                    sampling_executed = True

                    sampling_module, sampling_dp, sampling_batch, _ = self._get_sampling_contract(model_id)
                    assert sampling_module is not None
                    assert sampling_batch is not None
                    combined_params = format_sampling_params(sampling_params, sampling_batch)
                    max_prompt_len = max(int(prompt_lens[i]) for i in range(len(empty_slots)))
                    combined_prompt_tokens = torch.zeros(sampling_batch, max_prompt_len, dtype=torch.long)
                    for local_idx, slot in enumerate(empty_slots):
                        plen = int(prompt_lens[local_idx])
                        combined_prompt_tokens[slot, :plen] = prefill_ids[slot, :plen]

                    sampling_module.apply_prefill_state(
                        sampling_params=combined_params,
                        prompt_tokens=combined_prompt_tokens,
                        empty_slots=empty_slots,
                        replicate_seeds=False,
                    )

                    user_hidden = self.model[model_id].extract_last_tokens_batched_prefill(
                        logits,
                        last_token_idx,
                        padded_batch,
                        prefill_seq_len,
                        target_batch=sampling_batch,
                    )

                    sampling_trace_key = f"sampling_{prefill_seq_len}_{model_id}_{sampling_batch}_{sampling_dp}"
                    if enable_trace_current_prompt:
                        if self.trace_id_prefill_sampling[sampling_trace_key] is None:
                            (
                                s_trace_id,
                                s_trace_output,
                                s_trace_input,
                            ) = self._capture_trace_prefill_sampling(model_id, sampling_batch)
                            self.trace_id_prefill_sampling[sampling_trace_key] = s_trace_id
                            self.trace_output_prefill_sampling[sampling_trace_key] = s_trace_output
                            self.trace_input_prefill_sampling[sampling_trace_key] = s_trace_input

                        s_trace_input = self.trace_input_prefill_sampling[sampling_trace_key]
                        user_hidden_host = user_hidden.cpu()
                        ttnn.copy_host_to_device_tensor(user_hidden_host, s_trace_input)
                        ttnn.execute_trace(
                            self.model_args[model_id].mesh_device,
                            self.trace_id_prefill_sampling[sampling_trace_key],
                            cq_id=0,
                            blocking=False,
                        )
                        tt_tokens, tt_log_probs = self.trace_output_prefill_sampling[sampling_trace_key]
                    else:
                        batched_logits = self.model[model_id]._apply_norm_and_lm_head(user_hidden)
                        tt_tokens, tt_log_probs = self.model[model_id].sampling.sample(
                            batched_logits,
                            enable_trace=False,
                        )

                    ttnn.synchronize_device(self.model[model_id].mesh_device)

                    tokens_host = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)
                    log_probs_host = (
                        ttnn.to_torch(ttnn.get_device_tensors(tt_log_probs)[0]).reshape(-1)
                        if tt_log_probs is not None
                        else None
                    )
                    for local_idx, slot in enumerate(empty_slots):
                        output_tokens[slot] = tokens_host[slot]
                        if log_probs_host is not None:
                            output_log_probs[slot] = log_probs_host[slot]
                else:
                    if return_hidden_states:
                        # Embedding models: trace returns hidden states; extract last-token hidden per slot
                        slot_hidden_list = []
                        for local_idx, slot in enumerate(empty_slots):
                            user_hidden = logits[slot : slot + 1, :, :, :]
                            slot_hidden = self.model[model_id].process_hidden_states_after_prefill_trace(
                                user_hidden, last_token_idx[slot]
                            )
                            slot_hidden_list.append((slot, slot_hidden, last_token_idx[slot]))
                        ttnn.synchronize_device(self.model[model_id].mesh_device)
                        dim = self.model[model_id].args.dim
                        for slot, slot_hidden, lt_idx in slot_hidden_list:
                            slot_hidden_torch = ttnn.to_torch(ttnn.get_device_tensors(slot_hidden)[0]).float()
                            pos = int(lt_idx % 32)
                            out = slot_hidden_torch[0, 0, pos, :dim].clone()
                            if out.device.type != "cpu":
                                out = out.cpu()
                            output_tensor[slot] = out
                    else:
                        for local_idx, slot in enumerate(empty_slots):
                            user_logits = logits[slot : slot + 1, :, :, :]
                            if self._v25_split_logits_model_id == model_id:
                                # V2.5 full-DEFAULT path: slice+lm_head under DEFAULT.
                                # DO NOT restore DECODE here — the caller's untilize and
                                # subsequent ops also run under DEFAULT until next decode.
                                # DECODE restored at start of _decode_forward_trace_text.
                                logger.info(f"[V2.5] split-logits batched: slot={slot}, slice+lm_head under DEFAULT; restore deferred to decode")
                                _logits = self.model[model_id].slice_logits_for_prefill_trace(user_logits, last_token_idx[slot])
                                _logits = self.model[model_id]._apply_norm_and_lm_head(_logits)
                                # _v25_split_logits_model_id stays set; first slot done
                            else:
                                _logits = self.model[model_id].process_logits_after_prefill_trace(
                                    user_logits, last_token_idx[slot]
                                )
                            _logits = ttnn.to_layout(
                                _logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                            )
                            output_tensor[slot] = self.model[model_id].process_output_prefill(
                                _logits.cpu(), last_token_idx=(last_token_idx[slot] % 32)
                            )
                break

            # Non-batched prefill path
            if enable_trace_current_prompt:
                if return_hidden_states:
                    hidden_states = self.model[model_id].process_hidden_states_after_prefill_trace(
                        logits, last_token_idx
                    )
                    prefill_results.append(
                        {
                            "idx": idx,
                            "model_id": model_id,
                            "last_token_idx": last_token_idx,
                            "hidden_states": hidden_states.cpu(blocking=False),
                        }
                    )
                    continue
                else:
                    if self._v25_split_logits_model_id == model_id:
                        # V2.5 full-DEFAULT path (Layer 20 v3):
                        #   Run ttnn.slice AND norm+lm_head under DEFAULT manager.
                        #   Layer 20 v3 skips all L1-CB clash validation under DEFAULT,
                        #   so lm_head and untilize don't spuriously clash with GlobalCB.
                        #   DO NOT restore DECODE here — ttnn.untilize (below) also needs
                        #   DEFAULT manager. After untilize, logits.cpu(blocking=True) is used
                        #   (blocking=True, not blocking=False) to flush all device work while
                        #   still under DEFAULT manager (no stall-group restrictions). DECODE
                        #   is restored AFTER the blocking cpu() completes, before the sync loop.
                        logger.info(f"[V2.5] split-logits: slice+norm+lm_head under DEFAULT; untilize+blocking-cpu also under DEFAULT; restore after cpu (model_id={model_id})")
                        logits = self.model[model_id].slice_logits_for_prefill_trace(logits, last_token_idx)
                        logits = self.model[model_id]._apply_norm_and_lm_head(logits)
                        # Note: _v25_split_logits_model_id stays set; DECODE restored before sync loop
                        logger.info(f"[V2.5] split-logits: norm+lm_head done under DEFAULT; pending DECODE restore before sync loop")
                    else:
                        logits = self.model[model_id].process_logits_after_prefill_trace(logits, last_token_idx)
            else:
                if return_hidden_states:
                    raise NotImplementedError("return_hidden_states=True requires enable_trace=True")

            if sampling_enabled:
                tt_tokens, tt_log_probs = self.model[model_id].sampling.sample(
                    logits,
                    enable_trace=False,
                )
                prefill_results.append(
                    {
                        "idx": idx,
                        "model_id": model_id,
                        "last_token_idx": last_token_idx,
                        "logits": [
                            tt_tokens.cpu(blocking=False),
                            tt_log_probs.cpu(blocking=False) if tt_log_probs is not None else None,
                        ],
                        "sampling": sampling_enabled,
                    }
                )
            else:
                logits = ttnn.untilize(logits, use_multicore=True)
                # V2.5 path: use blocking=True cpu() so that the DMA completes while still
                # under DEFAULT manager (no stall-group restrictions from the decode sub-device
                # manager). Restore DECODE AFTER this blocking transfer, BEFORE the sync loop.
                # Under DEFAULT the stall group is the full device; persistent sender/receiver
                # kernels have already exited (prefetcher.stop() ran at the end of the prior
                # decode step inside the trace), so the blocking cpu() completes quickly.
                _v25_active = (self._v25_split_logits_model_id == model_id)
                prefill_results.append(
                    {
                        "idx": idx,
                        "model_id": model_id,
                        "last_token_idx": last_token_idx,
                        "logits": logits.cpu(blocking=_v25_active),
                        "sampling": sampling_enabled,
                        "v25_blocking": _v25_active,
                    }
                )

        # V2.5: Restore DECODE manager BEFORE the sync loop.
        # All slice + lm_head + untilize ran under DEFAULT (no GlobalCB clash, no assertion).
        # The logits.cpu(blocking=True) above already completed the DMA, so synchronize_device
        # in the results loop is NOT needed for V2.5 results — just restore DECODE here.
        if self._v25_split_logits_model_id != -1:
            logger.info(
                f"[V2.5] restoring DECODE manager (blocking cpu done, model_id={self._v25_split_logits_model_id})"
            )
            self._v25_restore_prefetcher(self._v25_split_logits_model_id)
            self._v25_split_logits_model_id = -1

        if len(prefill_results) > 0:
            for elem_idx, res in enumerate(prefill_results):
                idx = res["idx"]
                last_token_idx = res["last_token_idx"]
                model_id = res["model_id"]
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                last_token_idx_relative = last_token_idx - num_cached_tokens
                if not res.get("v25_blocking"):
                    # V2.5 path: cpu(blocking=True) already flushed the DMA; skip synchronize.
                    # Non-V2.5 path: cpu(blocking=False) was used; must synchronize here.
                    ttnn.synchronize_device(self.model[model_id].mesh_device)

                if "hidden_states" in res:
                    output_tensor[idx] = self.model[model_id].process_output_prefill_hidden_states(
                        res["hidden_states"], last_token_idx=(last_token_idx_relative % 32)
                    )
                elif res["sampling"]:
                    tt_tokens = res["logits"][0]
                    tt_log_probs = res["logits"][1]
                    tokens_host = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)[
                        (
                            last_token_idx % 32
                        )  # TODO: Check if here should be used last_token_idx_relative instead of last_token_idx
                    ]
                    if isinstance(tt_log_probs, LogProbsResult):
                        log_probs_host = tt_log_probs.extract_user(last_token_idx % 32)
                    elif tt_log_probs is not None:
                        log_probs_host = ttnn.to_torch(ttnn.get_device_tensors(tt_log_probs)[0]).reshape(-1)[
                            (last_token_idx % 32)
                        ]  # TODO: Check if here should be used last_token_idx_relative instead of last_token_idx
                    else:
                        log_probs_host = None
                    output_tokens[idx] = tokens_host
                    if log_probs_host is not None:
                        output_log_probs[idx] = log_probs_host
                else:
                    output_tensor[idx] = self.model[model_id].process_output_prefill(
                        res["logits"], last_token_idx=(last_token_idx_relative % 32)
                    )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        if sampling_executed:
            return output_tokens, reformat_logprobs(output_log_probs, batch_size)
        else:
            return output_tensor

    def prefill_forward_single_user_text(
        self,
        tokens,  # New tokens to prefill (without the cached tokens), padded by get_padded_prefill_len()
        page_table,  # Cached and new pages
        user_id,
        last_token_idx,  # Last token index of the full prompt, including the cached tokens
        kv_cache=None,
        model_id=-1,
        num_cached_tokens: int = 0,
        batch_size=1,
        **kwargs,
    ):
        seq_len = tokens.shape[-1]
        use_chunked_prefill = seq_len > self.model_args[model_id].max_prefill_chunk_size
        use_prefix_caching = num_cached_tokens > 0
        if use_chunked_prefill or use_prefix_caching:
            """
            Chunked prefill requires paged attention. There are some strange constraints which we must meet:
             - page_table, which is used in SDPA, must match batch size of inputs, which is 1. This is because SDPA
             checks that page table batch dim matches input batch dim. Therefore we must slice the page table for the current user.
             - page_table must also have enough entries in each chunk, so it will be padded with zeros if necessary.
             - chunked_page_table is the slice of the page table for the current chunk. This is used by paged_fill_cache
             to keep it otherwise unaware that it is operating on a chunk.
             - due to the above point, we must always set user_id to 0 for chunked prefill.
            """
            assert page_table is not None, "page_table must be provided for chunked prefill"
            assert kv_cache is not None, "kv_cache must be provided for chunked prefill"
            assert last_token_idx is not None and last_token_idx < seq_len + num_cached_tokens, (
                f"last_token_idx must be provided and less than seq_len + num_cached_tokens: "
                f"last_token_idx={last_token_idx}, seq_len={seq_len}, num_cached_tokens={num_cached_tokens}"
            )

            if use_chunked_prefill:
                # If chunked prefill (more than one chunk is needed), we want to use the maximum chunk size.
                chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args[model_id].max_prefill_chunk_size)
            else:
                # Otherwise we only have one chunk.
                chunk_size = seq_len

            last_token_idx_in_seq = last_token_idx - num_cached_tokens  # Excluding the cached tokens
            block_size = get_block_size(kv_cache)
            last_token_idx_in_chunk = last_token_idx_in_seq % chunk_size
            # Calculate which chunk contains the last_token_idx
            last_chunk_start = (last_token_idx_in_seq // chunk_size) * chunk_size
            page_table_user = page_table[user_id : user_id + 1, :]
            # Pad page table to match number of blocks in seq_len
            num_padding_blocks = num_blocks_in_seq(seq_len + num_cached_tokens, block_size) - page_table_user.shape[1]
            page_table_user_padded = torch.cat(
                [page_table_user, torch.zeros(1, num_padding_blocks, dtype=torch.int32)], dim=-1
            )
            CHUNK_USER_ID = 0

            for chunk_start in range(num_cached_tokens, num_cached_tokens + seq_len, chunk_size):
                # These are absolute, i.e. including the cached tokens
                chunk_end = chunk_start + chunk_size
                # These are relative, i.e. excluding the cached tokens
                chunk_start_relative = chunk_start - num_cached_tokens
                chunk_end_relative = chunk_end - num_cached_tokens
                assert chunk_end <= num_cached_tokens + seq_len, (
                    f"chunk_end should be less or equal to "
                    f"num_cached_tokens + seq_len. "
                    f"Got: chunk_end={chunk_end}, "
                    f"num_cached_tokens={num_cached_tokens}, seq_len={seq_len}"
                )

                # Select tokens for the current chunk.
                # Cached tokens were already excluded (not part of the input),
                # so using relative indexes.
                chunk_tokens = tokens[:, chunk_start_relative:chunk_end_relative]

                # Select pages for the current chunk.
                # Cached pages must be skipped as well,
                # so using absolute indexes.
                chunk_page_table = page_table_user_padded[:, chunk_start // block_size : chunk_end // block_size]

                chunk_inputs = self.model[model_id].prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                    batch_size=batch_size,
                    user_id=CHUNK_USER_ID,
                    **kwargs,
                )
                (
                    chunk_prefill_input,
                    chunk_rot_mats_global_prefill,
                    chunk_rot_mats_local_prefill,
                    page_table_tt,
                    chunk_page_table_tt,
                ) = chunk_inputs

                tt_logits = self.model[model_id].ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats_global=chunk_rot_mats_global_prefill,
                    rot_mats_local=chunk_rot_mats_local_prefill,
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=(last_token_idx_in_chunk // 32) * 32,
                    kv_cache=kv_cache,
                    batch_size=batch_size,
                    **kwargs,
                )

                if chunk_start_relative == last_chunk_start:
                    return tt_logits
                else:
                    del tt_logits
        else:
            inputs = self.model[model_id].prepare_inputs_prefill(
                tokens,
                page_table=page_table,
                batch_size=batch_size,
                user_id=user_id,
                **kwargs,
            )
            prefill_input, rot_mats_global_prefill, rot_mats_local_prefill, page_table_tt, _ = inputs

            tt_logits = self.model[model_id].ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global_prefill,
                rot_mats_local=rot_mats_local_prefill,
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=-1 if batch_size > 1 else (last_token_idx // 32) * 32,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
            return tt_logits

    # Note: This function is called by vLLM
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params: SamplingParams = None,  # Should be None if not greedy decoding / sampling on device.
        reset_batch=False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap=None,
        **kwargs,
    ):
        mode_switched = False
        if self.mode != Mode.DECODE:
            self.mode = Mode.DECODE
            mode_switched = True

        # Switch to decode mode for prefetcher to reintialize sub devices
        for i in range(len(self.model)):
            self.model[i].switch_mode(Mode.DECODE)
        # Tenstorrent-p1 (V2.5): after switch_mode(Mode.DECODE) the DECODE sub-device
        # manager is active.  Mark that DECODE mode has been entered so _easy_trace_prefill
        # knows to suspend the prefetcher (V2.5) before any subsequent prefill trace capture.
        # Unlike Option-B, we do NOT release DEFAULT-captured traces here — V2.5 replays
        # them under DEFAULT manager (after suspending the prefetcher), avoiding both the
        # GlobalCB L1 clash and the per-device stream-0 counter mismatch.
        if not self._prefill_decode_mode_entered:
            _any_prefetcher_v2 = any(getattr(m, "prefetcher", None) is not None for m in self.model)
            if _any_prefetcher_v2:
                self._prefill_decode_mode_entered = True
                logger.info(
                    "[V2.5] DECODE manager activated — subsequent prefill trace captures "
                    "will suspend prefetcher (DEFAULT manager) via V2.5 before capture."
                )



        sampling_on_device = sampling_params is not None
        split_sampling_enabled = bool(self.enable_split_sampling and sampling_on_device)
        self._set_sampling_trace_mode(split_sampling_enabled)

        B = tokens.shape[0]

        tokens = torch.chunk(tokens, self.data_parallel, 0)
        start_pos = torch.chunk(start_pos, self.data_parallel, 0)
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None
        sampling_params_list = None
        if sampling_params is not None:
            # sampling_dp may differ from data_parallel for models that internally
            # shard users across mesh rows (users_row_sharded) — each row samples
            # 32 users independently, so sampling params must be chunked by the
            # number of rows even though data_parallel=1 for the forward pass.
            sampling_dp_values = [getattr(self.model[i], "sampling_dp", 1) for i in range(self.data_parallel)]
            assert (
                len(set(sampling_dp_values)) == 1
            ), f"All model instances must have the same sampling_dp, got {sampling_dp_values}"
            # NOTE: This assumes data_parallel and sampling_dp are mutually exclusive
            # (one is always 1). If a future model needs both DP>1 and row-sharded
            # sampling, this should become data_parallel * sampling_dp_values[0].
            sampling_dp = max(self.data_parallel, sampling_dp_values[0])

            sampling_params_list = chunk_sampling_params(sampling_params, sampling_dp)

            prompt_chunks = (
                torch.chunk(prompt_tokens, sampling_dp, 0) if prompt_tokens is not None else [None] * sampling_dp
            )
            output_chunks = (
                torch.chunk(output_tokens, sampling_dp, 0) if output_tokens is not None else [None] * sampling_dp
            )

            for i in range(self.data_parallel):
                sampling_module = getattr(self.model[i], "sampling", None)
                assert sampling_module is not None, "Sampling module not found in model for sampling on device."
                assert (
                    sampling_dp % self.data_parallel == 0
                ), f"sampling_dp ({sampling_dp}) must be divisible by data_parallel ({self.data_parallel})"
                cpm = sampling_dp // self.data_parallel
                start = i * cpm
                model_chunks = sampling_params_list[start : start + cpm]

                model_prompt = (
                    torch.cat([c for c in prompt_chunks[start : start + cpm] if c is not None], 0)
                    if prompt_tokens is not None
                    else None
                )
                model_output = (
                    torch.cat([c for c in output_chunks[start : start + cpm] if c is not None], 0)
                    if output_tokens is not None
                    else None
                )

                sampling_module.apply_decode_state(
                    model_chunks,
                    reset_batch=reset_batch,
                    prompt_tokens=model_prompt,
                    output_tokens=model_output,
                )
                # Apply slot remap from condense before advancing seeds.
                if slot_remap is not None:
                    sm_bs = sampling_module.seed_manager.max_batch_size
                    rank_remap = slot_remap[i * sm_bs : (i + 1) * sm_bs]
                    sampling_module.seed_manager.apply_slot_remap(rank_remap)
                sampling_module.seed_manager.get_new_values()

        decode_kwargs = {
            "current_pos": start_pos,
            "tokens": tokens,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "sampling_on_device": sampling_on_device,
        }

        if enable_trace:
            tt_decode_output = self._decode_forward_trace_text(**decode_kwargs, reset_batch=mode_switched)
        else:
            tt_decode_output = self._decode_forward_no_trace_text(**decode_kwargs)

        if read_from_device:
            to_host = self.read_decode_output(tt_decode_output)
            return self.process_decode_output_host(to_host, is_tokens=(sampling_params is not None))
        return tt_decode_output

    def _decode_forward_no_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
    ):
        """
        Performs text decode step.
        Returns tt_logits on device
        """
        tt_output = []
        tt_tokens = []
        tt_current_pos = []
        tt_rot_mat_idxs = []
        tt_page_table = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            model_i = self.model[i]
            (
                tt_tokens_i,
                tt_current_pos_i,
                tt_rot_mat_idxs_i,
                tt_page_table_i,
            ) = model_i.prepare_inputs_decode(tokens[i], current_pos[i], user_page_table)
            tt_tokens.append(tt_tokens_i)
            tt_current_pos.append(tt_current_pos_i)
            tt_rot_mat_idxs.append(tt_rot_mat_idxs_i)
            tt_page_table.append(tt_page_table_i)

        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            tt_logits_i, tt_log_probs_i = self.model[i].ttnn_decode_forward(
                tt_tokens[i],
                tt_current_pos[i],
                rot_mat_idxs=tt_rot_mat_idxs[i],
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                sampling_on_device=sampling_on_device,
                # Tenstorrent-p1 (Layer 15): compile run skips all_gather_async entirely
                # to avoid the multi-chip fabric deadlock (see model.py ttnn_decode_forward).
                # The compile-run logits are discarded; correctness only matters in trace replay.
                use_barrier_semaphore=False,
            )
            tt_output.append((tt_logits_i, tt_log_probs_i))

        return tt_output

    def _capture_decode_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
    ):
        """
        Captures a trace for the decode_forward method.
        """

        # Compile run
        self._decode_forward_no_trace_text(
            tokens,
            current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_on_device=sampling_on_device,
        )
        # Tenstorrent-p1 (Layer 15): synchronize before trace capture.
        # With the DRAM prefetcher active (3-sub-device DECODE manager), the compile
        # run dispatches sender (sub_device_0) + compute workers (sub_device_2) programs
        # asynchronously (use_barrier_semaphore=False returns raw TILE logits without
        # waiting).  Without synchronization, the compile-run GlobalCB sender may still
        # be streaming data into the trace-capture run's GlobalCB, causing a deadlock:
        #   - Two concurrent senders fight for the same GlobalCB circular buffer
        #   - end_mesh_trace finish() issues dispatch_wait(45264) which blocks dispatch_d
        #   - dispatch_d can't process trace-capture sender programs → workers starve → hang
        # Fix: synchronize the DECODE sub-device manager (stall group = [sub_device_2])
        # after compile run to drain compute workers.  The stall group waits for
        # sub_device_2 worker count = 45264, which implies sub_device_0 sender also
        # completed (GlobalCB protocol: sender only finishes after all layers acked).
        # This is safe with warm kernel cache (~37 ms per decode step, not 10+ min).
        for i in range(self.data_parallel):
            ttnn.synchronize_device(self.model_args[i].mesh_device)
        logger.info("Done Compiling Model")

        # Get inputs ready for trace run
        device_inputs = []
        tt_out_trace = []
        trace_ids = {}
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None

            host_inputs = self.model[i].prepare_decode_inputs_host(
                tokens[i], current_pos[i], page_table=user_page_table
            )

            device_inputs_i = copy_host_to_device(host_inputs, mesh_device=self.model_args[i].mesh_device)
            device_inputs.append(device_inputs_i)

        for i in range(self.data_parallel):
            sampling_module = getattr(self.model[i], "sampling", None)
            split_enabled = (
                sampling_on_device
                and sampling_module is not None
                and getattr(sampling_module, "enable_internal_trace", False)
            )
            trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
            trace_ids[i] = trace_id
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            prefetcher_active = getattr(self.model[i], "prefetcher", None) is not None
            tt_out_trace.append(
                self.model[i].ttnn_decode_forward(
                    *device_inputs[i],
                    kv_cache=user_kv_cache,
                    sampling_on_device=sampling_on_device,
                    capture_sampling_trace=split_enabled,
                    # Tenstorrent-p1 (Layer 15): gather-outside-trace strategy.
                    # With the DRAM prefetcher active, all three paths for the logit
                    # all-gather are illegal inside trace capture:
                    #   (a) all_gather_async: host-side CCL semaphore writes → TT_FATAL
                    #   (b) ttnn.all_gather: calls finish() → 3-sub-device deadlock in
                    #       compile run (workers wait for GlobalCB → sender waits for ACKs)
                    #   (c) ttnn.all_gather during trace capture: also triggers host-side
                    #       write_shard_to_device internally → TT_FATAL confirmed in testing
                    # Pass use_barrier_semaphore=False so model.py skips gather+untilize
                    # and returns raw TILE logits as the trace output.
                    # _decode_forward_trace_text applies gather+untilize after execute_trace.
                    use_barrier_semaphore=not prefetcher_active,
                )
            )
            ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)

            if split_enabled:
                sampling_module.capture_trace(logits=tt_out_trace[i], tt_out_tok=device_inputs[i][0])
        logger.info("Done Capturing Decode Trace")

        return trace_ids, tt_out_trace, *device_inputs

    def _decode_forward_trace_text(
        self, tokens, current_pos, page_table=None, kv_cache=None, sampling_on_device=False, reset_batch=False
    ):
        """
        Run decode forward text with tracing
        """
        # Tenstorrent-p1 (V2.5): restore prefetcher state after a prefill-interrupted decode.
        # _v25_decode_restore_pending is set True by _easy_trace_prefill when the prefetcher
        # was suspended (switched to DEFAULT manager) for prefill trace capture/replay.
        # Restore here — BEFORE trace capture or inputs reset — so that
        # _capture_decode_trace_text runs under the DECODE manager (GlobalCB active).
        if self._v25_decode_restore_pending:
            logger.info("[V2.5] Decode restore (safety net): restoring prefetcher state after prefill suspend")
            for model_id in range(self.data_parallel):
                self._v25_restore_prefetcher(model_id)
            self._v25_decode_restore_pending = False
            logger.info("[V2.5] Decode restore: complete — DECODE manager re-activated")
        # Safety net: if split-logits restore was never consumed (unexpected code path),
        # restore DECODE manager now before decode starts.
        if self._v25_split_logits_model_id != -1:
            logger.info(f"[V2.5] decode start: restoring DECODE manager (deferred from prefill split-logits, model_id={self._v25_split_logits_model_id})")
            self._v25_restore_prefetcher(self._v25_split_logits_model_id)
            self._v25_split_logits_model_id = -1

        # The trace is different depending on whether we are doing device sampling or not
        if not self.trace_ids_decode[sampling_on_device]:
            # WS-A.19: count decode-trace captures across the server's lifetime.
            # Expect 1 (or 2 if sampling_on_device flips). A monotonically-rising
            # counter would indicate per-request re-capture (suspect #3).
            self._wsa19_capture_ct = getattr(self, "_wsa19_capture_ct", 0) + 1
            if os.environ.get("SGLANG_TT_WSA19_TIMING", "0") == "1":
                logger.info(f"[WSA19] decode trace capture #{self._wsa19_capture_ct} "
                            f"(sampling_on_device={sampling_on_device}, "
                            f"reset_batch={reset_batch})")
            trace_ids, tt_out_trace, *device_inputs = self._capture_decode_trace_text(
                tokens, current_pos, page_table=page_table, kv_cache=kv_cache, sampling_on_device=sampling_on_device
            )
            self.trace_ids_decode[sampling_on_device] = trace_ids
            self.trace_inputs_decode[sampling_on_device] = device_inputs
            self.trace_output_decode[sampling_on_device] = tt_out_trace

        # reset inputs when mode switches from prefill to decode,
        # or when sampling_on_device changes (different trace has stale inputs)
        prev_sampling_on_device = getattr(self, "_prev_sampling_on_device", None)
        self._prev_sampling_on_device = sampling_on_device
        sampling_mode_changed = prev_sampling_on_device is not None and prev_sampling_on_device != sampling_on_device
        reset_inputs = reset_batch or not sampling_on_device or sampling_mode_changed
        if self.prev_page_table is None or any(
            not torch.equal(prev, curr) for prev, curr in zip(self.prev_page_table, page_table)
        ):
            # If the page table has changed, it means additional pages have been added or inputs are shuffled
            reset_inputs = True
            if page_table is not None:
                self.prev_page_table = tuple(pt.clone() for pt in page_table)

        if reset_inputs:
            for i in range(self.data_parallel):
                user_page_table = page_table[i] if page_table is not None else None
                host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)

                copy_host_to_device(
                    host_tensors=host_inputs_i,
                    device_tensors=self.trace_inputs_decode[sampling_on_device][i],
                )
        import time as _time_l15
        _t_exec0 = _time_l15.perf_counter()
        for i, trace_id in self.trace_ids_decode[sampling_on_device].items():
            ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
        _t_exec1 = _time_l15.perf_counter()
        outputs = self.trace_output_decode[sampling_on_device]

        # Tenstorrent-p1 (Layer 15): gather-outside-trace strategy.
        # When the DRAM prefetcher is active, trace capture skips all_gather + untilize
        # (use_barrier_semaphore=False path in model.py) to avoid:
        #   (a) host-side I/O from all_gather_async during trace capture, and
        #   (b) finish() deadlock from ttnn.all_gather during the compile run, and
        #   (c) host-side writes from ttnn.all_gather during trace capture (TT_FATAL).
        # The trace output is raw TILE logits (shape [1, 1, 1, vocab_size/num_devices]).
        # Apply all_gather + untilize here, OUTSIDE the trace, after execute_trace.
        _t_ag0 = _t_ag1 = _t_ut0 = _t_ut1 = _t_exec1
        _pref_active = False
        # Tenstorrent-p1 (Vector 3 debug): log first-ever check of the condition.
        _v3dbg_logged = getattr(self, "_v3dbg_pref_cond_logged", False)
        if not sampling_on_device:
            new_outputs = []
            for i in range(self.data_parallel):
                prefetcher = getattr(self.model[i], "prefetcher", None)
                _v3dbg_cond = prefetcher is not None and self.model_args[i].num_devices > 1
                if not _v3dbg_logged:
                    logger.info(
                        f"[V3-DBG] prefetcher={prefetcher is not None} num_devices={self.model_args[i].num_devices} "
                        f"_pref_active_cond={_v3dbg_cond} sampling_on_device={sampling_on_device}"
                    )
                    self._v3dbg_pref_cond_logged = True
                    _v3dbg_logged = True
                if _v3dbg_cond:
                    _pref_active = True
                    tt_logits, tt_log_probs = outputs[i]
                    num_links = 2 if self.model_args[i].is_galaxy else 1
                    _t_ag0 = _time_l15.perf_counter()
                    tt_logits = ttnn.all_gather(
                        tt_logits,
                        dim=3,
                        num_links=num_links,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=self.model_args[i].ccl_topology(),
                    )
                    _t_ag1 = _time_l15.perf_counter()
                    _t_ut0 = _t_ag1
                    tt_logits = ttnn.untilize(
                        tt_logits,
                        use_multicore=True,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        sub_core_grids=prefetcher.all_worker_cores_range_set,
                    )
                    _t_ut1 = _time_l15.perf_counter()
                    new_outputs.append((tt_logits, tt_log_probs))
                else:
                    new_outputs.append(outputs[i])
            outputs = new_outputs

        if _pref_active and os.environ.get("SGLANG_TT_PREFETCH_TIMING", "0") == "1":
            print(
                f"[L15-timing] exec_dispatch={1000*(_t_exec1-_t_exec0):.3f}ms "
                f"ag_dispatch={1000*(_t_ag1-_t_ag0):.3f}ms "
                f"ut_dispatch={1000*(_t_ut1-_t_ut0):.3f}ms "
                f"total_host_dispatch={1000*(_t_ut1-_t_exec0):.3f}ms",
                flush=True,
            )

        if sampling_on_device:
            new_outputs = []
            for i in range(self.data_parallel):
                sampling_module = getattr(self.model[i], "sampling", None)
                if sampling_module is None or not getattr(sampling_module, "enable_internal_trace", False):
                    new_outputs.append(outputs[i])
                    continue

                new_outputs.append(
                    sampling_module.sample(
                        logits=outputs[i],
                        tt_out_tok=self.trace_inputs_decode[sampling_on_device][i][0],
                    )
                )
            return new_outputs
        return outputs

    def _prefill_forward_single_user(
        self,
        vision_images,
        vision_mask,
        tokens,
        xattn_caches,
        user_id,
        total_len,
        prefill_len,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        model_id=-1,
    ):
        """
        Performs vision encode step then text prefill.
        Returns (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits)
        """
        B = tokens.shape[0]
        last_token_idx = prefill_len - 1

        text_only_inference = vision_images is None
        if not text_only_inference:
            (
                vision_tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
            ) = self.model[model_id].compute_vision_tokens_masks(
                batch_images=[vision_images],
                batch_masks=[vision_mask],
                total_len=total_len,
                prefill_len=prefill_len,
            )

            if cross_page_table is not None:
                num_vision_tokens = vision_tokens.shape[2]
                cross_page_table = self._get_prefill_user_page_table(cross_page_table, kv_cache, num_vision_tokens)
        else:
            (
                vision_tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
            ) = (None, None, None, None, None)

        if page_table is not None:
            page_table = self._get_prefill_user_page_table(page_table, kv_cache, prefill_len)

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            rot_mats,
            tt_page_table,
            tt_cross_page_table,
        ) = self.model[model_id].prepare_inputs_prefill(
            tokens,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            prefill_len=prefill_len,
            page_table=page_table,
            cross_page_table=cross_page_table,
            text_only_inference=text_only_inference,
        )

        tt_logits = self.model[model_id].ttnn_prefill_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            xattn_caches,
            rot_mats,
            user_id,
            vision_tokens,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            get_last_token=(last_token_idx // 32) * 32,
            cross_page_table=tt_cross_page_table,
            text_only_inference=text_only_inference,
        )

        del tt_page_table
        del tt_cross_page_table

        return (
            xattn_caches,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            tt_logits,
        )

    # Note: This function is called by vLLM
    def prefill_forward(
        self,
        vision_images,
        vision_masks,
        tokens,
        xattn_caches,
        total_lens,
        prompt_lens,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        empty_slots=None,
        **kwargs,
    ):
        if not self.model_args[0].is_llama_vision():
            logits = self.prefill_forward_text(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                pixel_values=vision_images,
                **kwargs,
            )

            return logits, None, None, None, None

        else:
            (
                output_logits,
                prefill_output_xattn_masks,
                prefill_output_full_text_row_masked_out_masks,
                decode_output_xattn_masks,
                decode_output_full_text_row_masked_out_masks,
            ) = self.prefill_forward_llama_vision(
                vision_images,
                vision_masks,
                tokens,
                xattn_caches,
                total_lens,
                prompt_lens,
                page_table=page_table,
                kv_cache=kv_cache,
                cross_page_table=cross_page_table,
                empty_slots=empty_slots,
            )

            return (
                output_logits,
                prefill_output_xattn_masks,
                prefill_output_full_text_row_masked_out_masks,
                decode_output_xattn_masks,
                decode_output_full_text_row_masked_out_masks,
            )

    # Note: This function is called by vLLM
    def prefill_forward_llama_vision(
        self,
        vision_images,
        vision_masks,
        tokens: torch.Tensor,
        xattn_caches,
        total_lens,
        prompt_lens,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        empty_slots=None,
    ):
        """
        Batched version of _prefill_forward_single_user for vision model.
        """
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        if cross_page_table is not None:
            assert isinstance(cross_page_table, torch.Tensor), "cross_page_table mush be torch.Tensor"

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size

        output_logits = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)

        out_list = []
        prefill_output_xattn_masks = []
        prefill_output_full_text_row_masked_out_masks = []
        decode_output_xattn_masks = []
        decode_output_full_text_row_masked_out_masks = []

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        for idx, user_id in enumerate(empty_slots):
            model_id = user_id // max_batch_size_per_model
            group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
            seq_len = int(prompt_lens[idx])

            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            user_page_table = page_table[idx : idx + 1] if page_table is not None else None
            user_cross_page_table = cross_page_table[idx : idx + 1] if kv_cache is not None else None
            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None
            model_xattn_cache = xattn_caches[model_id] if xattn_caches is not None else None

            (
                model_xattn_cache,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
                logits,
            ) = self._prefill_forward_single_user(
                vision_images=vision_images[idx],
                vision_mask=vision_masks[idx],
                tokens=tokens[idx : idx + 1, :seq_len],  # Keep batch dimension
                xattn_caches=model_xattn_cache,
                user_id=group_user_id,
                total_len=total_lens[idx],
                prefill_len=seq_len,
                page_table=user_page_table,
                kv_cache=model_kv_cache,
                cross_page_table=user_cross_page_table,
                model_id=model_id,
            )

            if xattn_caches is not None:
                xattn_caches[model_id] = model_xattn_cache

            out_list.append(logits)
            prefill_output_xattn_masks.append(prefill_cross_attention_masks)
            prefill_output_full_text_row_masked_out_masks.append(prefill_full_text_row_masked_out_mask)
            decode_output_xattn_masks.append(decode_cross_attention_masks)
            decode_output_full_text_row_masked_out_masks.append(decode_full_text_row_masked_out_mask)

        # We gather prefill output at the end of prefill to reduce unnecessary device sync
        for idx, user_id in enumerate(empty_slots):
            model_id = user_id // max_batch_size_per_model

            last_token_idx = prompt_lens[idx] - 1
            output_logits[idx] = self.model[model_id].process_output_prefill(
                out_list[idx].cpu(), 1, last_token_idx=(last_token_idx % 32)
            )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return (
            output_logits,
            prefill_output_xattn_masks,
            prefill_output_full_text_row_masked_out_masks,
            decode_output_xattn_masks,
            decode_output_full_text_row_masked_out_masks,
        )

    # Note: This function is called by vLLM
    def decode_forward_llama_vision(
        self,
        start_pos,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        enable_trace=True,
        read_from_device=True,
    ):
        B = tokens.shape[0]
        data_parallel = min(B, self.data_parallel)
        batch_per_device = B // data_parallel
        tokens = torch.chunk(tokens, self.data_parallel, 0)
        start_pos = torch.chunk(start_pos, self.data_parallel, 0)
        prefill_cross_attention_masks = [
            prefill_cross_attention_masks[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        prefill_full_text_row_masked_out_mask = [
            prefill_full_text_row_masked_out_mask[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        decode_cross_attention_masks = [
            decode_cross_attention_masks[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        decode_full_text_row_masked_out_mask = [
            decode_full_text_row_masked_out_mask[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None
        cross_page_table = (
            torch.chunk(cross_page_table, self.data_parallel, 0) if cross_page_table is not None else None
        )

        decode_kwargs = {
            "position_id": start_pos,
            "tokens": tokens,
            "prefill_cross_attention_masks": prefill_cross_attention_masks,
            "prefill_full_text_row_masked_out_mask": prefill_full_text_row_masked_out_mask,
            "decode_cross_attention_masks": decode_cross_attention_masks,
            "decode_full_text_row_masked_out_mask": decode_full_text_row_masked_out_mask,
            "xattn_caches": xattn_caches,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "cross_page_table": cross_page_table,
        }
        if enable_trace:
            tt_logits = self._easy_trace(**decode_kwargs)
        else:
            tt_logits = self._decode_forward_no_trace(**decode_kwargs)

        if read_from_device:
            to_host = self.read_decode_output(tt_logits)
            return self.process_decode_output_host(to_host)
        else:
            return tt_logits

    # Note: This function is called by vLLM
    def read_decode_output(self, tt_out, async_read=False):
        """
        Input tt_out is list of tuples of (tt_out_tok, tt_log_probs)
        tt_log_probs can be: ttnn.Tensor (old path), LogProbsResult (new path), or None.
        """

        def _read_logprobs(lp, blocking: bool = True):
            if lp is None:
                return None
            return lp.cpu(blocking=blocking)

        if not async_read:
            if isinstance(tt_out[0], tuple):
                return [(out[0].cpu(), _read_logprobs(out[1])) for out in tt_out]
            elif isinstance(tt_out[0], ttnn.Tensor):
                return [out.cpu() for out in tt_out]

        host_outputs = []
        read_events = []
        for i in range(self.data_parallel):
            if isinstance(tt_out[i], tuple):
                outputs = (
                    tt_out[i][0].cpu(blocking=False),
                    _read_logprobs(tt_out[i][1], blocking=False),
                )
                host_outputs.append(outputs)
            elif isinstance(tt_out[i], ttnn.Tensor):
                outputs = tt_out[i].cpu(blocking=False)
                host_outputs.append(outputs)

            read_events.append(ttnn.record_event(self.model[i].mesh_device, 0))

        return host_outputs, read_events

    # Note: This function is called by vLLM
    def process_decode_output_host(self, tt_out, is_tokens=False):
        """
        Converts the input ttnn host tensors to torch tensors.
        The input can be logits (if is_tokens=False) or tokens (if is_tokens=True).
        When the decode output includes logprobs:
           * Old path: a single logprobs tensor is converted to a torch tensor.
           * New path (LogProbsResult): the LogProbsResult is converted into a
            tuple of torch tensors (topk_lp, topk_idx), where each has shape [batch, top_k].
         Returns:
           * If using the old path: (logits, log_probs) where both are torch tensors
             concatenated across data-parallel ranks.
           * If any rank uses the new path: (logits, (topk_lp, topk_idx)), where
             logits, topk_lp, and topk_idx are torch tensors concatenated across
             data-parallel ranks.
        """
        from models.common.sampling.tt_log_probs import LogProbsResult

        max_batch_size_per_model = self.model_args[0].max_batch_size

        logits = []
        log_probs = []
        for i in range(self.data_parallel):
            if isinstance(tt_out[i], tuple):
                logits_i = self.model[i].process_output_decode(
                    tt_out[i][0], max_batch_size_per_model, S=1, is_tokens=is_tokens
                )
                lp = tt_out[i][1]
                if isinstance(lp, LogProbsResult):
                    # New path: convert LogProbsResult to torch (topk_lp, topk_idx) tuple.
                    #
                    # LogProbsResult contains device tensors of shape (1,1,32,32) — 32 users
                    # × 32 top-k logprobs — replicated across all devices in the mesh.
                    # However, for row-sharded sampling (sampling_dp > 1), each mesh row
                    # independently computes logprobs for its own 32 users, so the content
                    # differs per row even though the tensor is "replicated."
                    #
                    # We cannot use a mesh composer (ConcatMesh2dToTensor) because it would
                    # concatenate all 32 devices including 8 column replicas per row, giving
                    # 8× duplicated data. Instead:
                    #   - Row-sharded (sampling_dp > 1): pick one device per row (first in
                    #     each row), read its [32, 32] tensor, concatenate rows → [128, 32].
                    #   - Non-row-sharded (sampling_dp == 1): read from a single device.
                    lp_tensor = lp.topk_logprobs_host if lp.topk_logprobs_host is not None else lp.topk_logprobs
                    idx_tensor = lp.topk_indices_host if lp.topk_indices_host is not None else lp.topk_indices
                    sampling_dp = getattr(self.model[i], "sampling_dp", 1)
                    if sampling_dp > 1:
                        # Row-sharded: read one device per row and concatenate
                        rows, cols = self.mesh_device.shape
                        device_tensors_lp = ttnn.get_device_tensors(lp_tensor)
                        device_tensors_idx = ttnn.get_device_tensors(idx_tensor)
                        row_lps = []
                        row_idxs = []
                        for row in range(rows):
                            dev_idx = row * cols  # first device in this row
                            row_lp = ttnn.to_torch(device_tensors_lp[dev_idx])
                            row_lps.append(row_lp.reshape(-1, row_lp.shape[-1])[:max_batch_size_per_model])
                            row_idx = ttnn.to_torch(device_tensors_idx[dev_idx])
                            row_idxs.append(row_idx.reshape(-1, row_idx.shape[-1])[:max_batch_size_per_model])
                        topk_lp = torch.cat(row_lps, dim=0).float()
                        topk_idx = torch.cat(row_idxs, dim=0).to(torch.int32)
                    else:
                        # Non-row-sharded: read from first device only
                        device_tensors_lp = ttnn.get_device_tensors(lp_tensor)
                        device_tensors_idx = ttnn.get_device_tensors(idx_tensor)
                        topk_lp = (
                            ttnn.to_torch(device_tensors_lp[0])
                            .reshape(-1, device_tensors_lp[0].shape[-1])[:max_batch_size_per_model]
                            .float()
                        )
                        topk_idx = (
                            ttnn.to_torch(device_tensors_idx[0])
                            .reshape(-1, device_tensors_idx[0].shape[-1])[:max_batch_size_per_model]
                            .to(torch.int32)
                        )
                    logits.append(logits_i)
                    log_probs.append((topk_lp, topk_idx))
                elif lp is not None:
                    # Old path: single logprob tensor
                    log_probs_i = self.model[i].process_output_decode(
                        lp, max_batch_size_per_model, S=1, is_tokens=is_tokens, is_log_probs=True
                    )
                    logits.append(logits_i)
                    log_probs.append(log_probs_i)
                else:
                    logits.append(logits_i)
                    log_probs.append(torch.ones(logits_i.shape))
            elif isinstance(tt_out[i], ttnn.Tensor):
                logits_i = self.model[i].process_output_decode(
                    tt_out[i], max_batch_size_per_model, S=1, is_tokens=is_tokens
                )
                logits.append(logits_i)
                log_probs.append(torch.ones(logits_i.shape))
            else:
                raise ValueError(f"Invalid type of tt_out: {type(tt_out[i])}")

        # Check if any DP rank returned new-path tuples (topk_lp, topk_idx)
        has_topk = any(isinstance(lp, tuple) for lp in log_probs)
        if has_topk:
            # New path: all DP ranks should have tuples. For ranks that
            # returned a dummy tensor (e.g. sz=0), create matching dummy tuples.
            normalized = []
            for lp in log_probs:
                if isinstance(lp, tuple):
                    normalized.append(lp)
                else:
                    # Dummy: shape [B, 32] zeros to match tuple format
                    B = lp.shape[0]
                    normalized.append((torch.zeros(B, 32, dtype=torch.float32), torch.zeros(B, 32, dtype=torch.int32)))
            all_lp = torch.cat([lp[0] for lp in normalized], 0)
            all_idx = torch.cat([lp[1] for lp in normalized], 0)
            return (torch.cat(logits, 0), (all_lp, all_idx))
        return (torch.cat(logits, 0), torch.cat(log_probs, 0))

    def _decode_forward_no_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        Performs text decode step.
        Returns tt_logits on device
        """

        # forward_decode should be traced callable
        # decorator does compilation, capture, execute
        tt_h = []
        tt_xattn_mask = []
        tt_full_text_mask_expand_1NSH = []
        tt_full_text_mask_expand_11SD = []
        tt_position_id = []
        tt_rot_mats = []
        tt_page_table = []
        tt_cross_page_table = []

        for i in range(self.data_parallel):
            B, S = tokens[i].shape
            assert S == 1

            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rot_mats_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = self.model[i].prepare_inputs_decode(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id=position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            tt_h.append(tt_h_i)
            tt_xattn_mask.append(tt_xattn_mask_i)
            tt_full_text_mask_expand_1NSH.append(tt_full_text_mask_expand_1NSH_i)
            tt_full_text_mask_expand_11SD.append(tt_full_text_mask_expand_11SD_i)
            tt_position_id.append(tt_position_id_i)
            tt_rot_mats.append(tt_rot_mats_i)
            tt_page_table.append(tt_page_table_i)
            tt_cross_page_table.append(tt_cross_page_table_i)

        tt_logits = []
        tt_log_probs = []
        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            tt_logits_i, tt_log_probs_i = self.model[i].ttnn_decode_forward(
                tt_h[i],
                tt_xattn_mask[i],
                tt_full_text_mask_expand_1NSH[i],
                tt_full_text_mask_expand_11SD[i],
                xattn_cache,
                tt_position_id[i],
                tt_rot_mats[i],
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                cross_page_table=tt_cross_page_table[i],
            )
            tt_logits.append(tt_logits_i)
            tt_log_probs.append(tt_log_probs_i)

        return tt_logits, tt_log_probs

    def _capture_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        Captures a trace for the decode_forward method.
        """
        tt_h = []
        tt_xattn_mask = []
        tt_full_text_mask_expand_1NSH = []
        tt_full_text_mask_expand_11SD = []
        tt_position_id = []
        tt_rot_mats = []
        tt_page_table = []
        tt_cross_page_table = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rot_mats_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = self.model[i].prepare_inputs_decode(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id=position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            tt_h.append(tt_h_i)
            tt_xattn_mask.append(tt_xattn_mask_i)
            tt_full_text_mask_expand_1NSH.append(tt_full_text_mask_expand_1NSH_i)
            tt_full_text_mask_expand_11SD.append(tt_full_text_mask_expand_11SD_i)
            tt_position_id.append(tt_position_id_i)
            tt_rot_mats.append(tt_rot_mats_i)
            tt_page_table.append(tt_page_table_i)
            tt_cross_page_table.append(tt_cross_page_table_i)

        # Compile run
        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            # tt_logits_rm and tt_log_probs_rm unused later, no need to make a list
            tt_logits_rm, tt_log_probs_rm = self.model[i].ttnn_decode_forward(
                tt_h[i],
                tt_xattn_mask[i],
                tt_full_text_mask_expand_1NSH[i],
                tt_full_text_mask_expand_11SD[i],
                xattn_cache,
                tt_position_id[i],
                tt_rot_mats[i],
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                cross_page_table=tt_cross_page_table[i],
            )
        logger.info("Done Compiling Model")

        # Get inputs ready for trace run
        tt_h = []
        tt_xattn_mask = []
        tt_full_text_mask_expand_1NSH = []
        tt_full_text_mask_expand_11SD = []
        tt_position_id = []
        tt_rope_id = []
        tt_page_table = []
        tt_cross_page_table = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rope_id_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = self.model[i].prepare_decode_inputs_host(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rope_id_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = copy_host_to_device(
                (
                    tt_h_i,
                    tt_xattn_mask_i,
                    tt_full_text_mask_expand_1NSH_i,
                    tt_full_text_mask_expand_11SD_i,
                    tt_position_id_i,
                    tt_rope_id_i,
                    tt_page_table_i,
                    tt_cross_page_table_i,
                ),
                mesh_device=self.model_args[i].mesh_device,
            )

            tt_h.append(tt_h_i)
            tt_xattn_mask.append(tt_xattn_mask_i)
            tt_full_text_mask_expand_1NSH.append(tt_full_text_mask_expand_1NSH_i)
            tt_full_text_mask_expand_11SD.append(tt_full_text_mask_expand_11SD_i)
            tt_position_id.append(tt_position_id_i)
            tt_rope_id.append(tt_rope_id_i)
            tt_page_table.append(tt_page_table_i)
            tt_cross_page_table.append(tt_cross_page_table_i)

        tt_h_trace_input = tt_h

        tt_logits_rm = []
        tt_log_probs_rm = []
        trace_ids = {}
        # Do on-device transformations of inputs before forward
        for i in range(self.data_parallel):
            trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
            trace_ids[i] = trace_id
            B = tokens[i].shape[0]
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            (
                tt_h_transform,
                tt_rot_mats,
                tt_xattn_mask_transform,
                tt_full_text_mask_expand_1NSH_transform,
                tt_full_text_mask_expand_11SD_transform,
            ) = self.model[i].transform_decode_inputs_device(
                tt_h[i],
                tt_rope_id[i],
                tt_xattn_mask[i],
                tt_full_text_mask_expand_1NSH[i],
                tt_full_text_mask_expand_11SD[i],
                B=B,
            )

            tt_logits_rm_i, tt_log_probs_rm_i = self.model[i].ttnn_decode_forward(
                tt_h_transform,
                tt_xattn_mask_transform,
                tt_full_text_mask_expand_1NSH_transform,
                tt_full_text_mask_expand_11SD_transform,
                xattn_cache,
                tt_position_id[i],
                tt_rot_mats,
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                cross_page_table=tt_cross_page_table[i],
            )
            tt_logits_rm.append(tt_logits_rm_i)
            tt_log_probs_rm.append(tt_log_probs_rm_i)
            ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return (
            trace_ids,
            tt_logits_rm,
            tt_log_probs_rm,
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        )

    def _decode_forward_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        page_table,
        cross_page_table,
        trace_ids,
        trace_logits_rm,
        trace_h,
        trace_xattn_mask,
        trace_full_text_mask_expand_1NSH,
        trace_full_text_mask_expand_11SD,
        trace_position_id,
        trace_rope_id,
        trace_page_table,
        trace_cross_page_table,
    ):
        """
        Executes the trace for the decode_forward method but does not read back outputs.
        """
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_full_text_mask_expand_11SD,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ) = self.model[i].prepare_decode_inputs_host(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id=position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            copy_host_to_device(
                host_tensors=(
                    tt_h,
                    tt_xattn_mask,
                    tt_full_text_mask_expand_1NSH,
                    tt_full_text_mask_expand_11SD,
                    tt_position_id,
                    tt_rope_id,
                    tt_page_table,
                    tt_cross_page_table,
                ),
                device_tensors=(
                    trace_h[i],
                    trace_xattn_mask[i],
                    trace_full_text_mask_expand_1NSH[i],
                    trace_full_text_mask_expand_11SD[i],
                    trace_position_id[i],
                    trace_rope_id[i],
                    trace_page_table[i],
                    trace_cross_page_table[i],
                ),
            )
        for i, trace_id in trace_ids.items():
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_logits_rm

    def _easy_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if not hasattr(self, "trace_ids"):
            (
                trace_ids,
                tt_logits_rm,
                tt_log_probs_rm,
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_full_text_mask_expand_11SD,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ) = self._capture_trace(
                position_id,
                tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
                xattn_caches,
                page_table=page_table,
                kv_cache=kv_cache,
                cross_page_table=cross_page_table,
            )
            self.trace_ids = trace_ids
            self.trace_inputs = {
                "tt_h": tt_h,
                "tt_xattn_mask": tt_xattn_mask,
                "tt_full_text_mask_expand_1NSH": tt_full_text_mask_expand_1NSH,
                "tt_full_text_mask_expand_11SD": tt_full_text_mask_expand_11SD,
                "tt_position_id": tt_position_id,
                "tt_rope_id": tt_rope_id,
                "tt_page_table": tt_page_table,
                "tt_cross_page_table": tt_cross_page_table,
            }
            self.trace_outputs = {
                "tt_logits_rm": tt_logits_rm,
            }

        trace_logits_rm = self._decode_forward_trace(
            position_id,
            tokens,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            page_table,
            cross_page_table,
            self.trace_ids,
            self.trace_outputs["tt_logits_rm"],
            self.trace_inputs["tt_h"],
            self.trace_inputs["tt_xattn_mask"],
            self.trace_inputs["tt_full_text_mask_expand_1NSH"],
            self.trace_inputs["tt_full_text_mask_expand_11SD"],
            self.trace_inputs["tt_position_id"],
            self.trace_inputs["tt_rope_id"],
            self.trace_inputs["tt_page_table"],
            self.trace_inputs["tt_cross_page_table"],
        )

        return trace_logits_rm

    def generate(
        self,
        vision_images,
        vision_mask,
        prompt_tokens,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        # Do initial prefill
        prefill_len = len(prompt_tokens)
        total_len = prefill_len + max_gen_len  # Prepares mask for full length of output

        prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long).reshape(1, -1)  # B, S
        # Suboptimal to allocate caches every time
        model_id = 0
        xattn_caches = self.model[model_id].setup_cache(self.model_args[model_id].max_batch_size)
        (
            xattn_caches,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            logits,
        ) = self._prefill_forward_single_user(
            vision_images,
            vision_mask,
            prompt_tokens_tensor,
            xattn_caches,
            user_id=0,
            total_len=total_len,
            prefill_len=prefill_len,
            model_id=model_id,
        )

        last_token_idx = prefill_len - 1
        logits = self.model[model_id].process_output_prefill(logits.cpu(), 1, last_token_idx=(last_token_idx % 32))
        logits = logits.view(1, 1, self.model_args[model_id].vocab_size)

        prefill_output_xattn_masks = [[] for _ in range(self.data_parallel)]
        prefill_output_full_text_row_masked_out_masks = [[] for _ in range(self.data_parallel)]
        decode_output_xattn_masks = [[] for _ in range(self.data_parallel)]
        decode_output_full_text_row_masked_out_masks = [[] for _ in range(self.data_parallel)]

        prefill_output_xattn_masks[model_id].append(prefill_cross_attention_masks)
        prefill_output_full_text_row_masked_out_masks[model_id].append(prefill_full_text_row_masked_out_mask)
        decode_output_xattn_masks[model_id].append(decode_cross_attention_masks)
        decode_output_full_text_row_masked_out_masks[model_id].append(decode_full_text_row_masked_out_mask)

        def sample(logits):
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            decoder = self.tokenizer or self.processor
            return next_token, decoder.decode(next_token.tolist())

        next_token, text = sample(logits)

        yield TokenResult(
            token=next_token[0].item(),
            text=text,
        )

        for gen_idx in range(max_gen_len - 1):
            position_id = torch.tensor([prefill_len + gen_idx])
            next_token_tensor = next_token.reshape(1, 1)  # B, S

            logits = self.decode_forward_llama_vision(
                position_id,
                next_token_tensor,
                prefill_output_xattn_masks,
                prefill_output_full_text_row_masked_out_masks,
                decode_output_xattn_masks,
                decode_output_full_text_row_masked_out_masks,
                [xattn_caches],
                enable_trace=False,
            )

            if isinstance(logits, tuple):
                logits = logits[0]

            next_token, text = sample(logits)
            yield TokenResult(
                token=next_token[0].item(),
                text=text,
            )

    def chat_completion(
        self,
        messages,
        temperature=0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        model_id = 0
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model[model_id].configuration.max_seq_len:
            max_gen_len = self.model[model_id].configuration.max_seq_len - 1

        encoder = self.processor or self.tokenizer
        model_input = encoder.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        vision_images = extract_images_from_messages(messages) or None
        vision_mask = None
        if vision_images is not None:
            vision_mask = create_vision_mask(model_input["input_ids"][0], encoder.image_token_id) or None

        tokens = []

        stop_reason = None
        for result in self.generate(
            vision_images=vision_images,
            vision_mask=vision_mask,
            prompt_tokens=model_input["input_ids"][0],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)
            if result.text == "<|eot_id|>":
                stop_reason = StopReason.end_of_turn
            elif result.text == "<|eom_id|>":
                stop_reason = StopReason.end_of_message

        if stop_reason is None:
            stop_reason = StopReason.out_of_tokens

        decoder = self.tokenizer or self.processor
        message = decoder.decode(tokens, skip_special_tokens=True)

        return CompletionMessage(message)

    def text_completion(
        self,
        content,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        """Supports only vision models at the moment"""
        model_id = 0
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model[model_id].configuration.max_seq_len:
            max_gen_len = self.model[model_id].configuration.max_seq_len - 1

        vision_images = []
        image_token = getattr(self.processor, "image_token", None) or getattr(self.tokenizer, "image_token", None)
        text = encode_content(content, vision_images, image_token)
        vision_images = vision_images or None
        model_input = self.processor(text=text, images=vision_images, add_special_tokens=False)
        vision_mask = None
        if vision_images is not None:
            vision_mask = create_vision_mask(model_input["input_ids"][0], self.processor.image_token_id) or None

        tokens = []

        for result in self.generate(
            vision_images=vision_images,
            vision_mask=vision_mask,
            prompt_tokens=model_input["input_ids"],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)

        decoder = self.tokenizer or self.processor
        generation = decoder.decode(tokens, skip_special_tokens=True)

        return generation

    def _get_prefill_user_page_table(
        self,
        page_table,
        kv_cache,
        prefill_len,
        trace_enabled=False,
        prefill_seq_len=None,
        use_batched_prefill=False,
        user_id=None,
        padded_batch_size=None,
    ):
        block_size = get_block_size(kv_cache)

        if use_batched_prefill:
            batch_dim = padded_batch_size if padded_batch_size is not None else self.model_args[0].max_batch_size
            num_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
            page_table = page_table[:, :num_blocks]
            if trace_enabled:
                if page_table.shape[1] < num_blocks:
                    padding = torch.ones(page_table.shape[0], num_blocks - page_table.shape[1], dtype=torch.int32) * -1
                    page_table = torch.cat([page_table, padding], dim=1)
            padded_page_table = torch.ones(batch_dim, page_table.shape[1], dtype=torch.int32) * -1
            assert user_id is not None
            for i, user in enumerate(user_id):
                padded_page_table[user, :] = page_table[i, :]
            return padded_page_table
        else:
            # Non-batched: match reference main exactly
            num_blocks = 0
            if trace_enabled:
                num_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
            else:
                num_blocks = num_blocks_in_seq(prefill_len, block_size)
            if trace_enabled:
                if page_table.shape[1] < num_blocks:
                    padding = torch.ones(1, num_blocks - page_table.shape[1], dtype=torch.int32) * -1
                    page_table = torch.cat([page_table, padding], dim=1)
            return page_table[:, :num_blocks]

    ## Destructor

    def __del__(self):
        # Release all captured traces to prevent nanobind memory leaks
        # Traces must be released before closing the mesh device
        try:
            # Release prefill traces
            if hasattr(self, "trace_id_prefill"):
                for trace_key, trace_id in self.trace_id_prefill.items():
                    if trace_id is not None:
                        # Extract model_id from trace_key (format: "{prefill_seq_len}_{model_id}" or "{prefill_seq_len}_{model_id}_{batch_size}")
                        parts = trace_key.split("_")
                        model_id = int(parts[1]) if len(parts) >= 2 else 0
                        try:
                            ttnn.release_trace(self.model_args[model_id].mesh_device, trace_id)
                        except Exception:
                            pass  # Ignore errors during cleanup

            # Release prefill sampling traces
            if hasattr(self, "trace_id_prefill_sampling"):
                for trace_key, trace_id in self.trace_id_prefill_sampling.items():
                    if trace_id is not None:
                        parts = trace_key.split("_")
                        if parts and parts[0] == "sampling" and len(parts) >= 3:
                            m_id = int(parts[2])
                        else:
                            m_id = int(parts[-1]) if len(parts) >= 2 else 0
                        try:
                            ttnn.release_trace(self.model_args[m_id].mesh_device, trace_id)
                        except Exception:
                            pass

            # Release decode traces
            if hasattr(self, "trace_ids_decode"):
                for sampling_key, trace_ids_dict in self.trace_ids_decode.items():
                    if trace_ids_dict is not None:
                        for model_id, trace_id in trace_ids_dict.items():
                            if trace_id is not None:
                                try:
                                    ttnn.release_trace(self.model_args[model_id].mesh_device, trace_id)
                                except Exception:
                                    pass  # Ignore errors during cleanup

            # Release vision traces if present
            if hasattr(self, "trace_ids"):
                for model_id, trace_id in self.trace_ids.items():
                    if trace_id is not None:
                        try:
                            ttnn.release_trace(self.mesh_device, trace_id)
                        except Exception:
                            pass  # Ignore errors during cleanup
        except Exception:
            pass  # Ignore any errors during trace cleanup

        # Workaround for issue #19052
        if self.data_parallel > 1:
            for m in self.model:
                ttnn.close_mesh_device(m.mesh_device)

        if hasattr(super(Generator, self), "__del__"):
            super().__del__()


def create_submeshes(mesh_device, data_parallel):
    if not isinstance(mesh_device, ttnn.MeshDevice) or data_parallel == 1:
        return [mesh_device]

    num_rows, num_cols = mesh_device.shape
    num_devices = num_rows * num_cols
    assert num_devices % data_parallel == 0, f"Unsupported device split: {num_devices} devices, {data_parallel} groups"

    if num_rows == 8 and num_cols == 4 and num_cols % data_parallel == 0:
        submeshes = mesh_device.create_submeshes(ttnn.MeshShape(num_rows, num_cols // data_parallel))
        for submesh in submeshes:
            submesh.reshape(ttnn.MeshShape(1, num_devices // data_parallel))
        return submeshes

    return mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))
