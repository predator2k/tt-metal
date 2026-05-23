# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, TensorGroup

# ATTACK-2: env-gated logit magnitude probe. Logs (step, max_abs, has_nan,
# has_inf, top_token_id) for the first SGLANG_TT_LOGIT_PROBE_STEPS=N decode
# steps. Use this to distinguish "first replay is dirty" from "every replay
# is dirty", and to detect mode-collapse / corruption before sampler crashes.
_LOGIT_PROBE_STEPS_REMAINING = None
_LOGIT_PROBE_STEP_COUNTER = 0


def _logit_probe(label, result):
    """Inspect a decode-forward result and log magnitude / NaN stats.

    `result` may be a torch.Tensor (logits) or a tuple/list whose elements
    contain a torch.Tensor with `.float()` (the typical SGLang shape after
    process_decode_output_host).  Errors are swallowed — this is a probe.
    """
    global _LOGIT_PROBE_STEPS_REMAINING, _LOGIT_PROBE_STEP_COUNTER
    if _LOGIT_PROBE_STEPS_REMAINING is None:
        try:
            _LOGIT_PROBE_STEPS_REMAINING = int(os.getenv("SGLANG_TT_LOGIT_PROBE_STEPS", "0") or 0)
        except ValueError:
            _LOGIT_PROBE_STEPS_REMAINING = 0
    if _LOGIT_PROBE_STEPS_REMAINING <= 0:
        return
    _LOGIT_PROBE_STEP_COUNTER += 1
    try:
        candidate = result
        # Unwrap tuple/list.
        while isinstance(candidate, (list, tuple)) and len(candidate) > 0:
            candidate = candidate[0]
        if not isinstance(candidate, torch.Tensor):
            return
        t = candidate.detach().float()
        max_abs = float(t.abs().max().item()) if t.numel() else float("nan")
        has_nan = bool(torch.isnan(t).any().item())
        has_inf = bool(torch.isinf(t).any().item())
        # Top token id of the first user/slot.
        try:
            flat = t.reshape(-1, t.shape[-1])[0]
            top_id = int(flat.argmax().item())
        except Exception:
            top_id = -1
        logger.warning(
            f"[LOGIT-PROBE] {label} step={_LOGIT_PROBE_STEP_COUNTER} "
            f"shape={tuple(t.shape)} max_abs={max_abs:.3e} "
            f"has_nan={has_nan} has_inf={has_inf} top_id={top_id}"
        )
    except Exception as _exc:
        logger.warning(f"[LOGIT-PROBE] {label} step={_LOGIT_PROBE_STEP_COUNTER} probe FAILED: {_exc}")
    finally:
        _LOGIT_PROBE_STEPS_REMAINING -= 1


def allocate_sglang_kv_cache(kv_cache_shape, dtype, num_layers, dp_model: List[Transformer], tt_cache_path):
    """Allocate a per-layer paged KV cache list.

    For models with hybrid attention (Qwen3.5: ``layer_types`` interleaves
    ``linear_attention`` and ``full_attention``), this function honors
    ``model.args.layer_types`` and emits ``None`` for linear-attention layers
    — those layers don't have a KV cache (they keep their own recurrent
    state inside ``LinearAttentionBlock``). The per-layer list length is
    still ``num_layers`` so ``model.forward()`` can index by layer position.

    For pre-hybrid models (Llama, Qwen3, etc.), ``layer_types`` is None so
    every entry is a real paged KV pair — byte-exact with the prior
    behavior.
    """
    logger.warning("[TT-METAL-SGLANG-LOG] allocate_sglang_kv_cache called in generator")
    submesh_devices = [model.mesh_device for model in dp_model]
    kv_cache = []
    for mesh_idx, submesh in enumerate(submesh_devices):
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        # Hybrid-attention models stash per-layer types on model.args;
        # absent attribute means "every layer is full attention" (default).
        layer_types = getattr(dp_model[mesh_idx].args, "layer_types", None)
        kv_tt = []
        for layer_num in tqdm(range(num_layers), desc=f"Allocating TT kv caches for each layer (submesh {mesh_idx+1})"):
            # Skip KV allocation for linear-attention layers (they use a
            # host-side recurrent state managed by LinearAttentionBlock).
            if layer_types is not None and layer_types[layer_num] == "linear_attention":
                kv_tt.append(None)
                continue
            # Get the dtype for the kv cache based on the configured optimizations in the model
            if dp_model[mesh_idx].args.optimizations is not None:
                kv_cache_dtype = dp_model[mesh_idx].args.optimizations.get_tensor_dtype(
                    decoder_id=layer_num, tensor=TensorGroup.KV_CACHE
                )
            else:
                kv_cache_dtype = None
            # Set default to bfloat8_b when no optimizations are configured
            kv_cache_dtype = ttnn.bfloat8_b if kv_cache_dtype is None else kv_cache_dtype
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=submesh,
                    # TODO: this could be ShardTensorToMesh, removing the need for sglang to know about TP for num_kv_heads.
                    # Could affect other calculations which use TTCacheEngine.num_kv_heads, though.
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_cache_dtype,
                    # Separate cache files for K and V to avoid collision.
                    cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
                )
                for kv in ["k", "v"]
            ]

            kv_tt.append(kv_tt_i)
        kv_cache.append(kv_tt)
    return kv_cache


def initialize_sglang_text_transformer(
    hf_config,
    tt_data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
    optimizations=DecodersPrecision.performance,
    use_prefetcher=False,
):
    # Tenstorrent-p1: optional DRAM prefetcher (hides DRAM-to-SRAM weight
    # load latency behind compute). Opt-in via use_prefetcher=True; the
    # caller is expected to have already verified is_prefetcher_supported
    # for the (model, num_devices) tuple. Default False to keep prior
    # behavior unchanged.
    from models.tt_transformers.tt.prefetcher import Prefetcher, is_prefetcher_supported

    submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
    # Load model args, weights
    model_args = []
    prefetchers = []
    for submesh in submesh_devices:
        # Per-submesh prefetcher (one Prefetcher per ModelArgs); num_layers
        # is set after ModelArgs construction below.
        prefetcher = None
        if use_prefetcher:
            num_devs = submesh.get_num_devices()
            # Tenstorrent-p1: gate uses default ring_size=16 which fails for
            # Qwen3-8B/num_devs=2 (1.67MB > 850KB cap). Iterate the constructor's
            # legal_receiver_cores [1,2,3,4,6,8,10] (ring_sizes [8,16,24,32,48,64,80]) and
            # accept if ANY passes. nrc=4/6 (ring_size=32/48) unblock Qwen3-8B whose
            # QKV tiles=96 require ring_size divisible by 96 (32 and 48 both divide 96).
            if any(
                is_prefetcher_supported(hf_config._name_or_path, num_devs, ring_size=rs)
                for rs in (8, 16, 24, 32, 48, 64, 80)
            ):
                # SGLANG_TT_PREFETCHER_SKIP_WO/SKIP_WQKV/SKIP_W1/SKIP_W3/SKIP_W2:
                # each excludes one weight from the prefetcher per-layer queue.
                # Default 5 weights/layer (wqkv, wo, w1, w3, w2).
                import os as _os_pf
                _num_pref_tensors = 5
                for _flag in (
                    "SGLANG_TT_PREFETCHER_SKIP_WO",
                    "SGLANG_TT_PREFETCHER_SKIP_WQKV",
                    "SGLANG_TT_PREFETCHER_SKIP_W1",
                    "SGLANG_TT_PREFETCHER_SKIP_W3",
                    "SGLANG_TT_PREFETCHER_SKIP_W2",
                ):
                    if _os_pf.environ.get(_flag, "0") == "1":
                        _num_pref_tensors -= 1
                prefetcher = Prefetcher(submesh, num_tensors=_num_pref_tensors, num_layers=n_layers)
        prefetchers.append(prefetcher)
        model_args_i = ModelArgs(
            submesh,
            instruct=(
                "Instruct" in hf_config._name_or_path or "DeepSeek-R1-Distill-Llama-70B" in hf_config._name_or_path
            ),
            max_batch_size=max_batch_size // tt_data_parallel,
            optimizations=lambda model_args: optimizations(model_args.n_layers, model_args.model_name),
            max_seq_len=max_seq_len,
            prefetcher=prefetcher,
        )

        assert model_args_i.model_name.replace("-", "") in hf_config._name_or_path.replace(
            "-", ""
        ), f"The model specified in sglang ({hf_config._name_or_path}) does not match the model name ({model_args_i.model_name}) with model weights ({model_args_i.CKPT_DIR})."
        if n_layers is not None:
            model_args_i.n_layers = n_layers
        if prefetcher is not None:
            prefetcher.num_layers = model_args_i.n_layers

        model_args.append(model_args_i)

    state_dict = model_args[0].load_state_dict()

    tt_model = []
    for i, submesh in enumerate(submesh_devices):
        tt_model_i = Transformer(
            args=model_args[i],
            mesh_device=submesh,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args[i].weight_cache_path(dtype),
            use_paged_kv_cache=True,
            prefetcher=prefetchers[i],
        )
        tt_model.append(tt_model_i)

    return tt_model, model_args


class LlamaForCausalLM(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_sglang_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
        use_prefetcher: bool = False,
    ):
        hf_model_name = hf_config._name_or_path
        if (
            ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
            and mesh_device.get_num_devices() == 1
            and is_wormhole_b0()
        ):
            MAX_PROMPT_LEN = 32768
            if max_seq_len > MAX_PROMPT_LEN:
                raise ValueError(
                    f"TT-LLama8B and TT-Llama11B do not support max_model_len greater than {MAX_PROMPT_LEN} on N150 "
                    f"(received {max_seq_len}). Set --max_model_len to {MAX_PROMPT_LEN} or lower in sglang."
                )

        tt_model, model_args = initialize_sglang_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.from_string(optimizations)
            if optimizations is not None
            else DecodersPrecision.performance,
            use_prefetcher=use_prefetcher,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_sglang_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class QwenForCausalLM(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_sglang_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
        use_prefetcher: bool = False,
    ):
        tt_model, model_args = initialize_sglang_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.from_string(optimizations)
            if optimizations is not None
            else DecodersPrecision.performance,
            use_prefetcher=use_prefetcher,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        result = super().decode_forward(*args, **kwargs)
        _logit_probe("Qwen", result)
        return result

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_sglang_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class MistralForCausalLM(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_sglang_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
        use_prefetcher: bool = False,
    ):
        tt_model, model_args = initialize_sglang_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.from_string(optimizations)
            if optimizations is not None
            else DecodersPrecision.performance,
            use_prefetcher=use_prefetcher,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_sglang_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class GptOssForCausalLM(Generator):
    """GPT-OSS model for sglang integration"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_sglang_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
        use_prefetcher: bool = False,
    ):
        from models.demos.gpt_oss.tt.common import create_tt_model

        optimizations = (
            DecodersPrecision.from_string(optimizations) if optimizations is not None else DecodersPrecision.performance
        )

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        model = []
        state_dict = None

        for submesh in submesh_devices:
            # Use the existing create_tt_model function
            model_args_i, model_i, _, state_dict = create_tt_model(
                mesh_device=submesh,
                instruct=True,
                max_batch_size=max_batch_size // tt_data_parallel,
                optimizations=lambda model_args: optimizations(model_args.n_layers, model_args.model_name),
                max_seq_len=max_seq_len,
                paged_attention_config=None,
                dtype=ttnn.bfloat8_b,
                state_dict=state_dict,
                num_layers=n_layers,
                mesh_config=None,
                create_kv_cache=False,
            )

            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(ttnn.bfloat8_b)

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_sglang_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)
