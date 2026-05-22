# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup, num_to_corerange
from models.tt_transformers.tt.rope import qwen35_partial_rotary_head_perm


class Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        self.prefetcher = prefetcher
        self.TG = self.num_devices == 32
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.ccl_dtype = configuration.ccl_dtype
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size
        self.rms_norm_add_unit_offset = configuration.rms_norm_add_unit_offset
        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        self.use_qk_fused = configuration.use_qk_fused
        self.use_hf_rope = configuration.use_hf_rope
        self.arch_name = configuration.arch_name
        # TODO: Fix this once all-gather supports < tile_size
        if self.TG:
            weight = torch.zeros(1, 32, 8, 32)
            for i in range(32):
                col = i % 4  # This determines which group of 8 to select
                weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

            self.slice_mat = ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            )
            user_selection_matrix = torch.eye(8, 8)
            user_selection_matrix = torch.nn.functional.pad(user_selection_matrix, (0, 24), "constant", 0)  # (8, 32)
            user_selection_matrix = [user_selection_matrix] * 4
            user_selection_matrix = torch.block_diag(*user_selection_matrix)  # (32, 128)
            self.user_selection_matrix = ttnn.from_torch(
                user_selection_matrix,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        self.dtype = dtype

        self.max_seq_len = configuration.max_seq_len
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats
        self.is_sliding = (
            configuration.layer_types[layer_num] == "sliding_attention" if configuration.layer_types else False
        )
        self.sliding_window = configuration.sliding_window if self.is_sliding else None

        self.model_config = configuration.get_model_config()
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip

        # When prefetcher is enabled, use consistent dtypes across all layers to avoid
        # race conditions caused by different block sizes
        use_prefetcher = prefetcher is not None

        decoders_optimizations = self.args.decoders_optimizations
        self.activation_dtype = decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION, prefetcher=use_prefetcher
        )
        self.wqkv_dtype = decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WQKV, prefetcher=use_prefetcher
        )
        self.wo_dtype = decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WO, prefetcher=use_prefetcher
        )
        self.kv_cache_dtype = decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.KV_CACHE, prefetcher=use_prefetcher
        )
        self.li_qkv_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_DECODE, configuration=configuration
        )
        self.sdpa_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_DECODE, configuration=configuration
        )
        self.li_o_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_DECODE, configuration=configuration
        )
        self.sdpa_prefill_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_PREFILL, configuration=configuration
        )
        self.li_qkv_prefill_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_PREFILL, configuration=configuration
        )
        self.li_o_prefill_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_PREFILL, configuration=configuration
        )

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        # Select rotary embedding implementation for decode
        if self.use_hf_rope and self.use_qk_fused:
            raise NotImplementedError("Fused QK is not implemented for HF-style rope")
        if self.use_hf_rope:
            self.rotary_embedding_decode = self._hf_rope_decode
        elif self.use_qk_fused:
            self.rotary_embedding_decode = self._mllama_rope_fused_qk_decode
        else:
            self.rotary_embedding_decode = self._mllama_rope_decode

        # Select rotary embedding implementation for prefill
        if self.use_hf_rope:
            self.rotary_embedding_prefill = self._hf_rope_prefill
        else:
            self.rotary_embedding_prefill = self._mllama_rope_prefill

        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Initialize bias tensors as None
        self.wqkv_bias_decode = None
        self.wqkv_bias_prefill = None

        # Create combined QKV bias if present in state dict
        if f"{wq_str}.bias" in state_dict:
            qkv_bias = torch.concat(
                [
                    torch.concat(
                        [
                            torch.chunk(state_dict[f"{wq_str}.bias"], configuration.num_devices)[i],
                            torch.chunk(state_dict[f"{wk_str}.bias"], configuration.num_devices)[i],
                            torch.chunk(state_dict[f"{wv_str}.bias"], configuration.num_devices)[i],
                        ],
                        dim=-1,
                    )
                    for i in range(configuration.num_devices)
                ],
                dim=-1,
            )
            # Prefill can use broadcasting on the bias add so wants a 1d tensor
            self.wqkv_bias_prefill = ttnn.as_tensor(
                qkv_bias,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("wqkv_bias_prefill_sharded"),
            )
            # as_tensor returns (32, dim) which is incorrect, this reshape updates the padded size to the correct size
            self.wqkv_bias_prefill = ttnn.reshape(
                self.wqkv_bias_prefill,
                (1, 1, 1, self.wqkv_bias_prefill.shape[-1]),
                (1, 1, self.wqkv_bias_prefill.shape[-2], self.wqkv_bias_prefill.shape[-1]),
            )

            # Broadcasting does not seem to be supported inside execute_trace so expand to the whole batch size
            # Create a list of bias tensors for each multiple of tile_size up to max_batch_size
            self.wqkv_bias_decode = []
            for batch_size in range(
                configuration.tile_size,
                configuration.tile_padded_batch_rows + configuration.tile_size,
                configuration.tile_size,
            ):
                qkv_bias_decode = qkv_bias.unsqueeze(0).expand(batch_size, -1)
                bias_tensor = ttnn.as_tensor(
                    qkv_bias_decode,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                    cache_file_name=cache_name(f"wqkv_bias_decode_sharded_{batch_size}"),
                )
                self.wqkv_bias_decode.append(bias_tensor)

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices_per_group == 0
        assert self.n_kv_heads % self.num_devices_per_group == 0
        assert configuration.qkv_size % self.num_devices_per_group == 0
        assert configuration.dim % self.num_devices_per_group == 0

        # WS-A.3: when attn_output_gate=True (e.g. Qwen3.5), the loaded q_proj
        # weight has shape [2 * n_heads * head_dim, hidden] in per-head
        # interleaved layout [q_h0 | gate_h0 | q_h1 | gate_h1 | ...] (each block
        # is head_dim rows). Plain torch.chunk(..., num_devices, dim=0) still
        # works to split heads across devices (each device gets its local
        # heads' q+gate pairs), but downstream attention math wants q and gate
        # to be CONTIGUOUSLY grouped — first all q-heads then all gate-heads —
        # so a single ttnn.reshape can split them after nlp_create_qkv_heads.
        # We deinterleave here, ONCE, at weight-load time.
        self.attn_output_gate = getattr(configuration, "attn_output_gate", False)

        # WS-A.6: partial_rotary_factor < 1 path (Qwen3.5). The TT
        # ``rotary_embedding_hf`` kernel hardcodes ``rotate_half`` at the
        # head_dim/2 boundary, but HF Qwen3.5 partial RoPE rotates only the
        # first ``rotary_dim`` dims and pairs (i, i + rotary_dim/2) inside that
        # slice. We permute the per-head rows of Q and K projection weights so
        # the rotated upper-half lands at TT index ``i + head_dim/2``, making
        # the kernel's bound rotate_half match HF's pair indices. cos/sin are
        # zero-padded (cos=1, sin=0) on the pass-through slots so those
        # positions become identity through the rope kernel. q_norm / k_norm
        # weights are permuted with the same scheme. V is untouched (no RoPE).
        # When ``partial_rotary_factor == 1.0`` (every other model) this branch
        # is a no-op and the wq/wk path is byte-equivalent.
        self.partial_rotary_factor = float(getattr(configuration, "partial_rotary_factor", 1.0))
        if self.partial_rotary_factor < 1.0:
            rotary_dim = int(self.head_dim * self.partial_rotary_factor)
            self._qk_rope_perm = qwen35_partial_rotary_head_perm(self.head_dim, rotary_dim)
        else:
            self._qk_rope_perm = None

        def _permute_qk_rows_per_head(weight: torch.Tensor, n_heads_per_block: int) -> torch.Tensor:
            """Apply self._qk_rope_perm to the per-head row blocks of a
            ``[n_heads_per_block * head_dim, hidden]`` weight tensor. No-op if
            partial_rotary is disabled. Returns a fresh tensor (does not mutate
            ``state_dict``).
            """
            if self._qk_rope_perm is None:
                return weight
            hidden = weight.shape[-1]
            # Reshape per-head, permute along head_dim row dim, then flatten.
            w = weight.view(n_heads_per_block, self.head_dim, hidden)
            w = w[:, self._qk_rope_perm, :]
            return w.reshape(n_heads_per_block * self.head_dim, hidden).contiguous()

        # wqkv DRAM-sharded width per device:
        #   - non-gated:   (n_heads + 2*n_kv_heads) * head_dim / num_devices
        #   - gated (Qwen3.5): (2*n_heads + 2*n_kv_heads) * head_dim / num_devices
        # (encoded as configuration.qkv_size already includes the gate factor).
        wqkv_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim, configuration.qkv_size // configuration.num_devices
        )

        qkv_list = []
        for i in range(self.num_devices_per_group):
            # Chunk weights
            wq_selected = torch.chunk(state_dict[f"{wq_str}.weight"], self.num_devices_per_group, dim=0)[i]
            wk_selected = torch.chunk(state_dict[f"{wk_str}.weight"], self.num_devices_per_group, dim=0)[i]
            wv_selected = torch.chunk(state_dict[f"{wv_str}.weight"], self.num_devices_per_group, dim=0)[i]

            # WS-A.3: deinterleave [q_h | gate_h] interleaved -> [all_q | all_gate]
            # PER DEVICE, so the final wqkv columns lay out as
            # [q_0,...,q_local_N-1, g_0,...,g_local_N-1, k_0,...,v_local_K-1].
            # After the QKV matmul + nlp_create_qkv_heads(num_heads=2*N_local),
            # the resulting Q tensor's leading "heads" are q-heads followed by
            # gate-heads, so a single reshape splits them.
            if self.attn_output_gate:
                # wq_selected shape: [n_local_heads * 2 * head_dim, hidden]
                # per-head layout [q_h0, g_h0, q_h1, g_h1, ...]
                hidden = wq_selected.shape[-1]
                wq_selected = wq_selected.view(self.n_local_heads, 2, self.head_dim, hidden)
                # split q and gate, concat along head-row dim with all-q first
                wq_q = wq_selected[:, 0, :, :].reshape(self.n_local_heads * self.head_dim, hidden)
                wq_g = wq_selected[:, 1, :, :].reshape(self.n_local_heads * self.head_dim, hidden)
                # WS-A.6: permute Q rows (not gate — gate is multiplied
                # element-wise after attention and is not rotated).
                wq_q = _permute_qk_rows_per_head(wq_q, self.n_local_heads)
                wq_selected = torch.cat([wq_q, wq_g], dim=0)
            else:
                # WS-A.6: permute Q rows for non-gated path (unused by Qwen3.5
                # in P1; future-proof for other partial-rotary models).
                wq_selected = _permute_qk_rows_per_head(wq_selected, self.n_local_heads)
            # WS-A.6: permute K rows.
            wk_selected = _permute_qk_rows_per_head(wk_selected, self.n_local_kv_heads)

            # Transpose the selected chunks
            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=self.wqkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if self.TG else wqkv_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(3, 2) if self.TG else (2, 3), mesh_shape=configuration.cluster_shape
            ),
            cache_file_name=cache_name("wqkv_sharded_2d"),
        )

        def norm_reshard(x, norm, mode, norm_config):
            """Hack until RMSNorm supports height-sharded output config"""
            if mode == Mode.DECODE:
                mem_cfg = x.memory_config()
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=x.dtype)
            x = norm(x, mode, norm_config=norm_config)
            if mode == Mode.DECODE:
                x = ttnn.to_memory_config(x, mem_cfg, dtype=x.dtype)
            return x

        # WS-A.6: when partial_rotary_factor < 1.0, permute q_norm/k_norm
        # weights with the same per-head row permutation applied to wq/wk.
        # q_norm runs on the Q tensor BEFORE rope and outputs in the same
        # layout, so its scale weights must match the permuted Q layout. Done
        # once per layer construction; idempotent because we overwrite the same
        # state_dict key in place. cache files will reflect the permuted
        # values (cache path was already invalidated by the use_hf_rope
        # subdir suffix; see model_config.py:583).
        if self._qk_rope_perm is not None:
            if f"{q_norm_str}.weight" in state_dict:
                state_dict[f"{q_norm_str}.weight"] = state_dict[f"{q_norm_str}.weight"][
                    self._qk_rope_perm
                ].contiguous()
            if f"{k_norm_str}.weight" in state_dict:
                state_dict[f"{k_norm_str}.weight"] = state_dict[f"{k_norm_str}.weight"][
                    self._qk_rope_perm
                ].contiguous()

        if f"{q_norm_str}.weight" in state_dict:
            fn_q_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,  # we already prefix q_norm_str
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=q_norm_str,
                add_unit_offset=self.rms_norm_add_unit_offset,
                is_distributed=False,
                tt_ccl=self.tt_ccl,
            )
            self.q_norm = lambda x, mode, norm_config: norm_reshard(x, fn_q_norm, mode, norm_config)
        else:
            self.q_norm = lambda x, mode, norm_config: x

        if f"{k_norm_str}.weight" in state_dict:
            fn_k_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,  # we already prefix k_norm_str
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=k_norm_str,
                add_unit_offset=self.rms_norm_add_unit_offset,
                is_distributed=False,
                tt_ccl=self.tt_ccl,
            )
            self.k_norm = lambda x, mode, norm_config: norm_reshard(x, fn_k_norm, mode, norm_config)
        else:
            self.k_norm = lambda x, mode, norm_config: x

        # For ring topology we can use all gather matmul for wo
        self.use_fused_all_gather_matmul = self.args.use_fused_all_gather_matmul
        pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        wo_mem_config = configuration.create_dram_sharded_mem_config(
            (configuration.n_heads * configuration.head_dim) // configuration.num_devices, configuration.dim
        )

        def get_wo_mesh_mapper():
            if self.use_fused_all_gather_matmul or self.TG:
                return ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(2, 3),
                    mesh_shape=configuration.cluster_shape,
                )
            return ttnn.ShardTensorToMesh(self.mesh_device, dim=2)

        if self.prefetcher is not None:
            self.wo_sharded_ring = ttnn.as_tensor(
                pt_wo,
                dtype=self.wo_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=self.args.get_sharded_wo_ring_mem_config(),
                mesh_mapper=get_wo_mesh_mapper(),
                cache_file_name=(cache_name("wo_sharded_ring")),
            )

        def get_wo_memory_config():
            if self.use_fused_all_gather_matmul or self.TG:
                return ttnn.DRAM_MEMORY_CONFIG
            else:
                return wo_mem_config

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=self.wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=get_wo_memory_config(),
            mesh_mapper=get_wo_mesh_mapper(),
            cache_file_name=(
                cache_name("wo_width_sharded_2d") if (self.use_fused_all_gather_matmul or self.TG) else cache_name("wo")
            ),
        )
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        if configuration.query_pre_attn_scalar is not None:
            self.scale = configuration.query_pre_attn_scalar**-0.5
        else:
            self.scale = self.head_dim**-0.5

        # Insert the tensors into the prefetcher only in decode mode, we do not use prefetcher in prefill mode
        if self.prefetcher is not None:

            def register_weights():
                self.prefetcher.insert_tensor(self.wqkv)
                self.prefetcher.insert_tensor(self.wo_sharded_ring)

            self.prefetcher.register_callback(register_weights)

    def init_kv_cache(self, configuration, weight_cache_path):
        """
        Generates empty KV cache and pushed to device memory
        """

        if self.paged_attention_config:
            cache_k = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
        else:
            cache_k = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=self.kv_cache_dtype,
                layout=self.args.get_attn_weights_layout(),
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=(
                    f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                    if weight_cache_path and not configuration.dummy_weights
                    else None
                ),
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def to_qk_fused_memory_config(self, q_tensor: ttnn.Tensor, k_tensor: ttnn.Tensor):
        """
        Convert Q and K tensors to height-sharded memory layouts suitable for
        fused QK ops such as rotary_embedding_llama_fused_qk and the subsequent
        QK matmul/attention score computation.

        This function:
        - Infers the number of Q heads and KV heads from the input tensors
        - Shards Q and K along the batch dimension using HEIGHT sharding
        - Places Q and K on disjoint core regions to avoid overlap within sub_core_grids
        - Uses row-major shard orientation with explicit shard shapes

        The resulting memory layouts are compatible with fused attention
        kernels that expect Q and K to be distributed across separate
        core ranges while preserving per-head contiguity.

        Args:
            q_tensor (ttnn.Tensor):
                Query tensor with shape [..., batch, num_q_heads, head_dim].

            k_tensor (ttnn.Tensor):
                Key tensor with shape [..., batch, num_kv_heads, head_dim].

            sub_core_grids (ttnn.CoreRangeSet):
                The available core grids to place Q and K tensors on.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]:
                (q_tensor, k_tensor) converted to sharded memory configurations.
        """
        n_q_heads = q_tensor.shape[2]
        n_kv_heads = k_tensor.shape[2]
        q_batch = q_tensor.shape[1]
        k_batch = k_tensor.shape[1]
        assert q_batch == k_batch

        row_size = 8  # We assume a row size of 8 cores
        k_start_core = ttnn.CoreCoord(q_batch % row_size, q_batch // row_size)

        q_core_grid = ttnn.CoreRangeSet({num_to_corerange(q_batch)})
        k_core_grid = ttnn.CoreRangeSet({num_to_corerange(k_batch, start_core=k_start_core)})

        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(nearest_32(n_q_heads), self.head_dim),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_mem_config = ttnn.create_sharded_memory_config(
            shape=(nearest_32(n_kv_heads), self.head_dim),
            core_grid=k_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        q_tensor = ttnn.to_memory_config(q_tensor, q_mem_config)
        k_tensor = ttnn.to_memory_config(k_tensor, k_mem_config)
        return q_tensor, k_tensor

    def _mllama_rope_decode(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos):
        # Q Rotary Embeddings
        q_heads_1BQD = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre_rot_1BQD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )

        # K Rotary Embeddings
        k_heads_1BKD = ttnn.experimental.rotary_embedding_llama(
            k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )
        return q_heads_1BQD, k_heads_1BKD

    def _mllama_rope_fused_qk_decode(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos):
        q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD = self.to_qk_fused_memory_config(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD
        )

        q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
        )
        return q_heads_1BQD, k_heads_1BKD

    def _hf_rope_decode(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos):
        if q_heads_pre_rot_1BQD.dtype != ttnn.bfloat16:
            q_heads_pre_rot_1BQD = ttnn.typecast(q_heads_pre_rot_1BQD, dtype=ttnn.bfloat16)
        if k_heads_pre_rot_1BKD.dtype != ttnn.bfloat16:
            k_heads_pre_rot_1BKD = ttnn.typecast(k_heads_pre_rot_1BKD, dtype=ttnn.bfloat16)

        q_heads_1BQD = ttnn.experimental.rotary_embedding_hf(
            q_heads_pre_rot_1BQD,
            rot_mats[0],
            rot_mats[1],
            is_decode_mode=True,
        )
        k_heads_1BKD = ttnn.experimental.rotary_embedding_hf(
            k_heads_pre_rot_1BKD,
            rot_mats[0],
            rot_mats[1],
            is_decode_mode=True,
        )
        return q_heads_1BQD, k_heads_1BKD

    def _mllama_rope_prefill(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )

        return q_heads_1QSD, k_heads_1KSD

    def _hf_rope_prefill(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_hf(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            is_decode_mode=False,
        )

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_hf(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            is_decode_mode=False,
        )

        return q_heads_1QSD, k_heads_1KSD

    def forward_decode(self, x: ttnn.Tensor, current_pos, rot_mats=None, page_table=None, kv_cache=None) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """

        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        # When ring_size does not evenly divide qkv_tiles, QKV falls back to the
        # standard DRAM-sharded config (no prefetcher ring).  In that case do NOT
        # pass global_cb / sub_device_id; MLP weights still use the prefetcher.
        xqkv_fused_sharded = ttnn.linear(
            x,
            self.wqkv,
            memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.DECODE, self.prefetcher),
            program_config=self.args.get_attn_qkv_program_config(Mode.DECODE, 1, self.prefetcher),
            compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
            dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None else None,
            # QKV ring-gather matmul: x is sharded on receiver cores, so the
            # factory's subdevice_cores query must use receiver_sub_device_id
            # (not worker_sub_device_id) to avoid an empty CoreRangeSet when
            # intersecting x.shard_spec().grid with subdevice_cores.
            sub_device_id=self.prefetcher.receiver_sub_device_id if self.prefetcher is not None else None,
        )
        # FIXME: File bug against dram-sharded matmuls with bias
        if self.wqkv_bias_decode:
            # select the bias tensor based on the number of tiles in the rows
            # WARNING: must not change the batch size between compiling and executing a trace
            num_tiles = int(math.ceil(xqkv_fused_sharded.shape[-2] / self.tile_size))
            xqkv_fused_sharded = xqkv_fused_sharded + self.wqkv_bias_decode[num_tiles - 1]

        ttnn.deallocate(x)
        qkv_all_reduce_mem_cfg = self.args.get_attn_qkv_all_reduce_output_mem_config(
            Mode.DECODE, list(self.mesh_device.shape)[1], self.prefetcher
        )
        xqkv_fused = tt_all_reduce(
            xqkv_fused_sharded,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            memory_config=qkv_all_reduce_mem_cfg
            if qkv_all_reduce_mem_cfg is not None
            else xqkv_fused_sharded.memory_config(),
            sharded=True,
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
            subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
        )
        if self.TG:
            # TODO: Slice the fused_query_key_value tensor get batch=8
            xqkv_fused = ttnn.matmul(
                self.slice_mat,
                xqkv_fused,
                dtype=ttnn.bfloat16,
                memory_config=self.args.get_attn_create_head_input_mem_config(Mode.DECODE),
            )
        else:
            # bfloat16 is required by nlp_create_qkv_heads_decode
            if self.prefetcher is None:
                # P3a.2 patch: our patched QKV matmul outputs DRAM-interleaved
                # (not sharded), so sharded_to_interleaved fails. Detect and
                # route to to_memory_config + typecast instead.
                if xqkv_fused_sharded.is_sharded():
                    xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
                else:
                    xqkv_fused = ttnn.to_memory_config(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG)
                    if xqkv_fused.dtype != ttnn.bfloat16:
                        xqkv_fused = ttnn.typecast(xqkv_fused, ttnn.bfloat16)
                ttnn.deallocate(xqkv_fused_sharded)
            else:
                xqkv_fused = xqkv_fused_sharded
        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, (1, 1, self.batch_size_per_device_group, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3])
        )

        ###
        # Reshape and rotary embeddings
        ###
        # WS-A.3: when attn_output_gate, wqkv has been built per-device as
        # [q (n_local_heads*head_dim) | gate (n_local_heads*head_dim) | k | v].
        # Tell nlp_create_qkv_heads to treat the Q section as 2*n_local_heads
        # "heads" of head_dim each; the kernel returns Q with shape
        # [1, batch, 2*n_local_heads, head_dim] where the first n_local_heads
        # are the real q-heads and the next n_local_heads are gate-heads.
        create_num_heads = self.n_local_heads * 2 if self.attn_output_gate else self.n_local_heads
        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=create_num_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE, self.prefetcher),
        )
        # Split Q tensor into [q | gate]. The sharded layout has shards of
        # [num_q_heads_padded, head_dim]; we move to DRAM_INTERLEAVED to do a
        # safe reshape+slice, then re-shard back to the original mem config.
        gate_heads_1BQD = None
        if self.attn_output_gate:
            q_mem_cfg = q_heads_pre_rot_1BQD.memory_config()
            q_il = ttnn.sharded_to_interleaved(q_heads_pre_rot_1BQD, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q_heads_pre_rot_1BQD)
            # shape: [1, batch, 2*n_local_heads, head_dim]; first n_local_heads
            # are q, next n_local_heads are gate.
            full_batch = q_il.shape[1]
            q_only = ttnn.slice(
                q_il,
                [0, 0, 0, 0],
                [1, full_batch, self.n_local_heads, self.head_dim],
            )
            gate_only = ttnn.slice(
                q_il,
                [0, 0, self.n_local_heads, 0],
                [1, full_batch, 2 * self.n_local_heads, self.head_dim],
            )
            ttnn.deallocate(q_il)
            # Re-shard q_only back to the original layout (matches what the
            # downstream RoPE / SDPA expect).
            q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_only, q_mem_cfg)
            ttnn.deallocate(q_only)
            # Keep gate in DRAM_INTERLEAVED; we use it post-SDPA, after
            # nlp_concat_heads_decode, where the tensor is in DRAM-interleaved
            # form anyway.
            gate_heads_1BQD = gate_only
        norm_config = self.args.get_norm_config("attn", Mode.DECODE, None)
        q_heads_pre_rot_1BQD = self.q_norm(q_heads_pre_rot_1BQD, mode=Mode.DECODE, norm_config=norm_config)
        k_heads_pre_rot_1BKD = self.k_norm(k_heads_pre_rot_1BKD, mode=Mode.DECODE, norm_config=norm_config)
        ttnn.deallocate(xqkv_fused)

        # Q, K Rotary Embeddings
        q_heads_1BQD, k_heads_1BKD = self.rotary_embedding_decode(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos
        )

        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)
        ###
        # KV update
        ###
        if kv_cache:
            keys = kv_cache[0]
            values = kv_cache[1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]

        # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
        # v_heads [seqlen, n_kv_heads, bsz, head_dim]
        # keys, [max_batch_size, n_kv_heads // configuration.num_devices, max_seq_len, head_dim]

        if self.use_qk_fused:
            ttnn.experimental.paged_fused_update_cache(
                keys, k_heads_1BKD, values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(
                keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)
        # NOTE: Varying the batch size will result in slightly different outputs.
        # For example, a prompt w/ 1 user vs, the same prompt repeated N times for N users, will produce different outputs
        # This is because the SDPA op in decode mode has different number of reductions depending on batch size
        # Which leads to slightly different outputs from attention (due to accumulated errors)
        sdpa_decode_prog_cfg = self.args.get_attn_sdpa_decode_program_config(self.prefetcher)
        # P3a.2 T2.2.Z+4: optional skip-self-attention mode (tree-mask
        # emulation for EAGLE verify chain). When self._skip_self_attention
        # is True, attention reads kv at [0, current_pos - 1) (excluding
        # the just-written K at current_pos), so the input token doesn't
        # leak into output logits via self-attention. This breaks the
        # chat-template "OkayOkay..." loop on EAGLE-3 verify.
        # Note: original blocking issue was ttnn.full() inside traced
        # decode; we use enable_trace=False in our verify path so this
        # is safe.
        if getattr(self, "_skip_self_attention", False):
            _one_t = ttnn.full(
                current_pos.shape,
                1,
                dtype=current_pos.dtype,
                device=current_pos.device(),
                memory_config=current_pos.memory_config(),
            )
            _attn_pos = ttnn.subtract(current_pos, _one_t)
        else:
            _attn_pos = current_pos
        if page_table is not None:
            attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=_attn_pos,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                program_config=sdpa_decode_prog_cfg,
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=_attn_pos,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                program_config=sdpa_decode_prog_cfg,
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,  # FIXME: why not L1 height sharded e.g. SCORES_BATCHED_MM_OUTPUT_MEMCFG?
            )

        ttnn.deallocate(q_heads_1BQD)
        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D,
            memory_config=self.args.get_attn_sdpa_output_mem_config(
                Mode.DECODE, self.batch_size_per_device_group, self.prefetcher
            ),
        )

        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_11BH,
            num_heads=self.n_local_heads,
            sub_core_grids=self.prefetcher.all_worker_cores_range_set if self.prefetcher is not None else None,
        )
        ttnn.deallocate(attn_output_11BH)
        ttnn.deallocate(attn_output_1G4D)

        # WS-A.3: apply attn_output_gate = sigmoid(gate) * attn_output before
        # o_proj. After nlp_concat_heads_decode, attn_output_cat has shape
        # [1, 1, batch, n_local_heads*head_dim]; gate_heads_1BQD has shape
        # [1, batch, n_local_heads, head_dim] in DRAM_INTERLEAVED. Reshape gate
        # and match memory_config of attn_output_cat for elementwise mul.
        if self.attn_output_gate and gate_heads_1BQD is not None:
            gate_flat = ttnn.reshape(
                gate_heads_1BQD,
                [1, 1, gate_heads_1BQD.shape[1], self.n_local_heads * self.head_dim],
            )
            gate_flat = ttnn.sigmoid(gate_flat)
            gate_flat = ttnn.to_memory_config(gate_flat, attn_output_cat.memory_config())
            attn_output_cat = ttnn.multiply(attn_output_cat, gate_flat)
            ttnn.deallocate(gate_flat)
            ttnn.deallocate(gate_heads_1BQD)

        if self.use_fused_all_gather_matmul or self.prefetcher is not None:
            attn_output_cat = ttnn.to_memory_config(
                attn_output_cat,
                self.args.get_attn_concat_heads_output_mem_config(Mode.DECODE, self.prefetcher),
            )

            # Fused AGMM only valid for ring topology
            if self.ccl_topology == ttnn.Topology.Ring and self.prefetcher is None:
                _, dense_out_sharded = ttnn.experimental.all_gather_matmul_async(
                    attn_output_cat,
                    self.wo,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    all_gather_core_grid_offset=(0, 4),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    num_links=self.model_config["ATTN_AGMM_CONFIG"]["num_links"],
                    memory_config_ag=self.args.get_attn_all_gather_output_mem_config(Mode.DECODE, None),
                    memory_config_mm=self.args.get_attn_dense_output_mem_config(Mode.DECODE, None),
                    program_config=self.args.get_attn_all_gather_matmul_program_config(Mode.DECODE, None),
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    chunks_per_sync=self.model_config["ATTN_AGMM_CONFIG"]["chunks_per_sync"],
                    num_workers_per_link=self.model_config["ATTN_AGMM_CONFIG"]["num_workers_per_link"],
                    num_buffers_per_channel=2,
                    subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
                )
            else:
                if self.prefetcher is not None:
                    all_gather_output = ttnn.experimental.all_gather_async(
                        attn_output_cat,
                        persistent_output_buffer=None,
                        dim=3,
                        multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                        num_links=1,
                        topology=self.ccl_topology,
                        memory_config=self.args.get_attn_all_gather_output_mem_config(Mode.DECODE, self.prefetcher),
                        barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                        chunks_per_sync=10,
                        num_workers_per_link=2,
                        num_buffers_per_channel=2,
                        subdevice_id=self.prefetcher.worker_sub_device_id,
                    )
                else:
                    # Standalone no-prefetcher path: use synchronous all_gather to avoid
                    # CCL semaphore slot reuse issue (see distributed_norm.py).
                    all_gather_output = ttnn.all_gather(
                        attn_output_cat,
                        dim=3,
                        num_links=1,
                        memory_config=self.args.get_attn_all_gather_output_mem_config(Mode.DECODE, self.prefetcher),
                        topology=self.ccl_topology,
                    )
                dense_out_sharded = ttnn.linear(
                    all_gather_output,
                    self.wo_sharded_ring if self.prefetcher is not None else self.wo,
                    memory_config=self.args.get_attn_dense_output_mem_config(Mode.DECODE, self.prefetcher),
                    program_config=self.args.get_attn_all_gather_matmul_program_config(Mode.DECODE, self.prefetcher),
                    compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
                    global_cb=self.prefetcher.global_cb if self.prefetcher is not None else None,
                    # dense_out ring-gather matmul: all_gather_output is sharded on receiver cores,
                    # so use receiver_sub_device_id (not worker_sub_device_id) to avoid empty CoreRangeSet.
                    sub_device_id=self.prefetcher.receiver_sub_device_id if self.prefetcher is not None else None,
                )
                ttnn.deallocate(all_gather_output)
            ttnn.deallocate(attn_output_cat)
            dense_out_sharded = ttnn.to_memory_config(
                dense_out_sharded,
                self.args.get_attn_dense_output_mem_config(Mode.DECODE, self.prefetcher),
            )
            return dense_out_sharded

        else:
            attn_output = tt_all_gather(
                attn_output_cat,
                self.mesh_device,
                self.tt_ccl,
                dim=2,
                cluster_axis=1,
                memory_config=self.args.get_attn_gather_users_mem_config(
                    Mode.DECODE, list(self.mesh_device.shape)[1], self.prefetcher
                ),
                sharded=True,
                subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
                # dtype=self.ccl_dtype,  # Running bf16 until we have SDPA output bfp8 df; otherwise we have two sharded to interleaved/interleaved to sharded conversions
            )
            if self.TG:
                attn_output = ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)
                # user_selection_matrix = [1, 1, 32, 128]
                # user_selection_matrix @ activation -> [1, 1, 32, 128] * [1, 1, 128, 2048] -> [1, 1, 32, 2048]
                attn_output = ttnn.matmul(
                    self.user_selection_matrix,
                    attn_output,
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )

            # TODO: Fix this once self.TG supports dram-sharded matmuls
            dense_out_sharded = ttnn.linear(
                attn_output,
                self.wo,
                core_grid=ttnn.CoreGrid(y=4, x=8) if self.TG else None,
                program_config=self.args.get_attn_wo_program_config(Mode.DECODE, 1, self.prefetcher),
                memory_config=self.args.get_attn_wo_output_mem_config(Mode.DECODE, self.prefetcher),
                dtype=ttnn.bfloat8_b if self.TG else None,
                compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None else None,
                sub_device_id=self.prefetcher.receiver_sub_device_id if self.prefetcher is not None else None,
            )

            ttnn.deallocate(attn_output_cat)

            # All reduce
            dense_out_reduced = tt_all_reduce(
                dense_out_sharded,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                dim=0 if (self.TG and self.hidden_size < 8192) else 3,
                topology=self.ccl_topology,
                memory_config=self.args.get_attn_all_reduce_output_mem_config(
                    Mode.DECODE, self.hidden_size, list(self.mesh_device.shape)[0], self.prefetcher
                ),
                sharded=True,
                dtype=self.ccl_dtype,
                use_composite=True if self.hidden_size == 8192 else False,
                subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
            )

            if not self.TG:
                dense_out_reduced = ttnn.to_memory_config(
                    dense_out_reduced, self.args.get_attn_dense_output_mem_config(Mode.DECODE, None)
                )

            return dense_out_reduced

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        # For batched prefill, x_11SH has shape [B, 1, S, H] where B is batch_size
        # concat before QKV matmul, then reshape back to batch after
        batch_size = x_11SH.shape[0]
        if batch_size > 1:
            # Concatenate batch dimension into sequence for matmul compatibility
            x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

        seq_len = x_11SH.shape[-2]
        original_seq_len = seq_len  # Track original for later unpadding
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        # Pad seq_len to nearest multiple of MAX_QKV_MM_SEQ_LEN if needed
        if seq_len > self.MAX_QKV_MM_SEQ_LEN and seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
            padded_seq_len = (
                (seq_len + self.MAX_QKV_MM_SEQ_LEN - 1) // self.MAX_QKV_MM_SEQ_LEN
            ) * self.MAX_QKV_MM_SEQ_LEN
            pad_len = padded_seq_len - seq_len
            x_11SH = ttnn.pad(x_11SH, padding=[(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)
            seq_len = padded_seq_len

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        if seq_len > 128:
            xqkv_fused = ttnn.experimental.minimal_matmul(
                x_11SH,
                self.wqkv,
                compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None),
            )
        else:
            xqkv_fused = ttnn.linear(
                x_11SH,
                self.wqkv,
                dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
                memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None),
                compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                program_config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None),
            )

        # FIXME: surely ttnn.linear bias should work?
        if self.wqkv_bias_prefill is not None:
            xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

        xqkv_fused = tt_all_reduce(
            xqkv_fused,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.ccl_dtype,
        )

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        # Slice back to original seq_len if we padded earlier
        if original_seq_len != seq_len:
            xqkv_fused = xqkv_fused[:, :, :original_seq_len, :]
            seq_len = original_seq_len

        if batch_size > 1:
            xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len // batch_size, -1])

        ttnn.deallocate(x_11SH)

        # WS-A.3: when attn_output_gate, the Q slot in wqkv is 2*n_local_heads
        # heads wide (q + gate). Tell nlp_create_qkv_heads to split it; we'll
        # peel off the gate heads after.
        create_num_heads_prefill = self.n_local_heads * 2 if self.attn_output_gate else self.n_local_heads
        # split qkv into heads
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=create_num_heads_prefill,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        norm_config = self.args.get_norm_config("attn", Mode.PREFILL, None)
        # P3a.2 patch: nlp_create_qkv_heads pads Q/K/V last-dim with GQA-fused
        # extra space (e.g. Qwen3-1.7B 8Q/4KV per device → padded 192 instead
        # of 128). RMSNorm validates against padded shape and rejects the
        # mismatch with gamma sized for head_dim. SDPA also rejects K/V hidden
        # dim mismatch. Strip all three back to head_dim before downstream ops.
        def _strip_to_head_dim(t):
            sh = list(t.shape)
            if sh[-1] != self.head_dim:
                return ttnn.slice(t, [0]*len(sh), sh[:-1] + [self.head_dim])
            return t
        q_heads_1QSD_pre_rot = _strip_to_head_dim(q_heads_1QSD_pre_rot)
        k_heads_1KSD_pre_rot = _strip_to_head_dim(k_heads_1KSD_pre_rot)
        v_heads_1VSD = _strip_to_head_dim(v_heads_1VSD)

        # WS-A.3: peel off gate-heads from the Q tensor. Output shape from
        # nlp_create_qkv_heads is [1, num_q_heads, seq, head_dim]; with gate
        # the first n_local_heads heads are q, the next n_local_heads are gate.
        gate_heads_prefill = None
        if self.attn_output_gate:
            full_shape = list(q_heads_1QSD_pre_rot.shape)  # [1, 2*N, S, D]
            q_only_p = ttnn.slice(
                q_heads_1QSD_pre_rot,
                [0, 0, 0, 0],
                [full_shape[0], self.n_local_heads, full_shape[2], full_shape[3]],
            )
            gate_only_p = ttnn.slice(
                q_heads_1QSD_pre_rot,
                [0, self.n_local_heads, 0, 0],
                [full_shape[0], 2 * self.n_local_heads, full_shape[2], full_shape[3]],
            )
            ttnn.deallocate(q_heads_1QSD_pre_rot)
            q_heads_1QSD_pre_rot = q_only_p
            gate_heads_prefill = gate_only_p

        q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)
        k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)

        ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###

        # Apply rotary embeddings using the selected implementation
        q_heads_1QSD, k_heads_1KSD = self.rotary_embedding_prefill(q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats)
        ttnn.deallocate(q_heads_1QSD_pre_rot)
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        # Fill KV-Cache
        if kv_cache:
            keys_BKSD, values_BKSD = kv_cache[0], kv_cache[1]
        else:
            keys_BKSD, values_BKSD = self.layer_past[0], self.layer_past[1]

        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=keys_BKSD.dtype)
        ttnn.deallocate(k_heads_1KSD)

        # sharding k_fill to deal with update_cache memory limitation
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and page_table is None:
            k_fill = ttnn.interleaved_to_sharded(k_heads_1KSD_8b, self.args.get_attn_kv_prefill_mem_config(seq_len))
        else:
            k_fill = k_heads_1KSD_8b

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=values_BKSD.dtype)

        ttnn.deallocate(v_heads_1VSD)

        # sharding v_fill to deal with update_cache memory limitation
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and page_table is None:
            v_fill = ttnn.interleaved_to_sharded(v_heads_1VSD_8b, self.args.get_attn_kv_prefill_mem_config(seq_len))
        else:
            v_fill = v_heads_1VSD_8b

        if self.TG:
            k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
            v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)
        if page_table is not None:
            # In the case that the tokens have been padded along the seq len dimension, we need to fill the cache with the unpadded k/v values.
            # Assume that the page table does not have padding, so we can use it to get the unpadded page len.
            block_size = keys_BKSD.shape[2]
            # If chunked prefill, use chunk_page_table if given, otherwise use page_table.
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table

        if batch_size > 1:
            # For batched prefill, loop over VALID users only and fill each user's cache separately
            # k_fill/v_fill have shape [padded_batch, n_kv_heads, seq_len_per_user, head_dim]
            # The paged_fill_cache kernel reads batch_idx_ptr[0] for all positions,
            # so we must call it once per user with their specific K/V slice
            #
            # IMPORTANT: user_id is a list of valid slot indices for batched prefill.
            # Empty slots have page_table entries of -1, so we must skip them to avoid
            # writing to invalid memory blocks.
            seq_len_per_user = k_fill.shape[2]
            page_len = fill_page_table.shape[1] * block_size

            # user_id is a list of valid slot indices (e.g., [0, 1, 2, ..., N-1] for N users)
            # Each slot index tells us which row in k_fill and page_table to use
            valid_slots = user_id if isinstance(user_id, (list, tuple)) else list(range(batch_size))

            for slot_idx in valid_slots:
                # Extract this slot's K/V slice: [1, n_kv_heads, seq_len_per_user, head_dim]
                k_user = k_fill[slot_idx : slot_idx + 1, :, :, :]
                v_user = v_fill[slot_idx : slot_idx + 1, :, :, :]

                # Slice to page length if needed (same as single-user path)
                k_user_sliced = k_user[:, :, :page_len, :] if page_len < seq_len_per_user else k_user
                v_user_sliced = v_user[:, :, :page_len, :] if page_len < seq_len_per_user else v_user

                # Fill cache for this specific slot with scalar batch_idx
                ttnn.experimental.paged_fill_cache(keys_BKSD, k_user_sliced, fill_page_table, batch_idx=slot_idx)
                ttnn.experimental.paged_fill_cache(values_BKSD, v_user_sliced, fill_page_table, batch_idx=slot_idx)
        elif page_table is not None:
            # Single user path with page_table
            page_len = fill_page_table.shape[1] * block_size
            k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
            v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill_sliced, fill_page_table, batch_idx=user_id)
        else:
            # Single user path without page_table
            ttnn.fill_cache(
                keys_BKSD,
                k_fill,
                user_id % self.batch_size_per_device_group,
            )
            ttnn.fill_cache(
                values_BKSD,
                v_fill,
                user_id % self.batch_size_per_device_group,
            )
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and page_table is None:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

        # SDPA
        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=self.activation_dtype or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        if chunk_start_idx is not None:
            if self.sliding_window is not None:
                raise NotImplementedError("Sliding window not supported for chunked prefill SDPA")
            attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q_heads_1QSD_8b,
                input_tensor_k=keys_BKSD,
                input_tensor_v=values_BKSD,
                page_table_tensor=page_table,
                chunk_start_idx=chunk_start_idx,
                compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, seq_len, chunk_start_idx, None),
            )
        else:
            # For batched prefill, the actual per-user seq_len is seq_len // batch_size
            # since the tensors have shape [batch_size, n_heads, seq_len_per_user, head_dim]
            sdpa_seq_len = seq_len // batch_size if batch_size > 1 else seq_len
            attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                is_causal=True,
                sliding_window_size=self.sliding_window,
                scale=self.scale,
                compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, sdpa_seq_len, None, None),
            )

        # deallocate keys and values
        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        # For single-user prefill, reshape to expected format for nlp_concat_heads
        # For batched prefill (batch_size > 1), skip this reshape - nlp_concat_heads handles [B, H, S, D]
        # IMPORTANT: Reshaping [B, H, S, D] to [1, H, B*S, D] BEFORE concat_heads would scramble data
        # because batch and sequence dimensions are separated by heads. Must reshape AFTER concat_heads.
        if batch_size == 1:
            attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])
        else:
            attn_output_1QSD = attn_output_84SD

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        # WS-A.3: apply attn_output_gate = sigmoid(gate) * attn_output before
        # o_proj in prefill. gate_heads_prefill has shape [1, n_local_heads, S, D]
        # and attn_output_11SH is [B, 1, S_per_user, H*D]. Concat gate heads
        # into a flat per-token vector matching attn_output_11SH layout.
        if self.attn_output_gate and gate_heads_prefill is not None:
            gate_concat = ttnn.experimental.nlp_concat_heads(
                gate_heads_prefill,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(gate_heads_prefill)
            if batch_size > 1:
                gate_concat = ttnn.reshape(gate_concat, [1, 1, seq_len, -1])
            gate_concat = ttnn.sigmoid(gate_concat)
            attn_output_11SH = ttnn.multiply(attn_output_11SH, gate_concat)
            ttnn.deallocate(gate_concat)

        # For batched prefill, reshape to concatenate batch dimension into sequence
        # This MUST happen AFTER nlp_concat_heads to preserve correct data layout
        # nlp_concat_heads outputs [B, 1, S_per_user, H*D], reshape to [1, 1, B*S, H*D]
        if batch_size > 1:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, 1, seq_len, -1])

        # reshaping long sequence to matmul fit on device
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        # Non fused All Gather Matmul
        if self.use_fused_all_gather_matmul:  # is true for Ring topology
            attn_output_11SH = ttnn.experimental.all_gather_async(
                attn_output_11SH,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
            dtype=self.activation_dtype or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.args.get_attn_wo_program_config(Mode.PREFILL, seq_len, None),
        )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # Reduce-scatter
        if not self.use_fused_all_gather_matmul:
            output_11SH = tt_all_reduce(
                output_11SH,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                dim=0 if self.TG else 3,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.ccl_dtype,
            )

        return output_11SH

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        if mode == Mode.PREFILL:
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )
        else:
            return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)

    def prefill_prepare_tensor_for_kv_cache(self, key_or_value_layer, user_id):
        tensor_copy = ttnn.clone(key_or_value_layer)
        # key_or_value_layer.deallocate(True)
        # Get all tensors from multi-device tensor
        tensors = ttnn.get_device_tensors(tensor_copy)
        # Get only tensors from specific column chips
        # Get every 4th tensor starting from user_id // 8
        single_column_tensors = tensors[user_id // self.batch_size_per_device_group :: 4]
        # Create multi-device tensor
        multi_device_tensor = ttnn.combine_device_tensors(tensors=single_column_tensors)

        return multi_device_tensor
