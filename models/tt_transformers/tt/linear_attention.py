# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LinearAttentionBlock — host-fallback GatedDeltaNet decoder block for Qwen3.5.

This is the *correctness-first* path for Qwen3.5's hybrid attention pattern
(layer types: ``linear_attention`` vs ``full_attention``). The full-attention
layers continue to use the existing TT-native ``TransformerBlock``; linear
layers run the GatedDeltaNet computation **on the host CPU in pure PyTorch**.

Why host fallback:
  * tt-metal has no GatedDeltaNet kernel.
  * Writing one is a separate workstream (multi-week, not WS-A.2 scope).
  * WS-A.2 only needs the smoke to advance past G3 so WS-A.3 (MRoPE) can land.

The MLP, residual-add, and the two RMSNorms (``attention_norm``, ``ffn_norm``)
run on device — same as ``TransformerBlock`` — so prefill/decode plumbing and
weight loading match the full-attention path. Only the ``self_attn`` substitute
is host-fallback.

Per-layer recurrent state lives on the host (``self._conv_state``,
``self._ssm_state``); the paged KV cache is bypassed for these layers. The
generator's ``allocate_sglang_kv_cache`` puts ``None`` entries in the per-layer
list for linear layers, and ``forward()`` here ignores its ``kv_cache=`` kwarg.

This block is intentionally slow (~50–200 ms per layer per decode step from the
host roundtrip). It is *not* a perf path; it exists so WS-A.3 can write the
MRoPE work without being blocked on a kernel that doesn't exist yet.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mlp import MLP


def _rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNormGated: gate-then-normalize. Mirrors SGLang's ``RMSNormGated`` with
    ``norm_before_gate=True`` semantics (SGLang's ``Qwen3_5GatedDeltaNet`` sets
    ``norm_before_gate=True`` at construction).

    Despite the parameter name, the math is:
        x = x * silu(gate)
        x = x / sqrt(mean(x**2) + eps) * weight
    """
    x = x * torch.nn.functional.silu(gate)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x * weight


class LinearAttentionBlock(LightweightModule):
    """Qwen3.5 linear-attention decoder block (host-fallback GatedDeltaNet).

    Shape conventions on host (per-decode-step, batch=B):
      * hidden    : ``[B, hidden_size]``
      * Q, K      : ``[B, num_k_heads, head_k_dim]``
      * V         : ``[B, num_v_heads, head_v_dim]``
      * z (gate)  : ``[B, num_v_heads, head_v_dim]``
      * a, b      : ``[B, num_v_heads]``
      * ssm_state : ``[B, num_v_heads, head_v_dim, head_k_dim]``
      * conv_state: ``[B, conv_dim, conv_kernel_size-1]`` (rolling window)

    Math (from SGLang fused_recurrent_gated_delta_rule_packed_decode_kernel):
        g    = -exp(A_log) * softplus(a + dt_bias)              # per head
        beta = sigmoid(b)                                       # per head
        h    = h * exp(g)                                       # state decay
        v   -= sum(h * k, dim=-1)                               # delta correction
        v   *= beta                                             # gate
        h   += outer(v, k)                                      # state update
        o    = sum(h * q, dim=-1)                               # output
    """

    SOFTPLUS_THRESHOLD = 20.0

    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        prefetcher=None,
        # The following are accepted-and-ignored to keep the constructor
        # signature compatible with TransformerBlock. Linear layers don't
        # use KV cache or rotation matrices.
        transformation_mats=None,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.layer_num = layer_num
        self.num_devices = args.num_devices
        self.hidden_size = args.dim
        self.dtype_torch = torch.bfloat16

        # ============================================================
        # Per-layer GatedDeltaNet shape parameters (from text_config)
        # ============================================================
        # ``ModelArgs`` doesn't carry these as named attributes (they only
        # appear in Qwen3.5), so dig them out of the raw text_config dict.
        # ``args._text_config`` is the dict ``model_config.py`` stashed at
        # parse time; fall back to standard Qwen3.5 defaults if missing.
        text_cfg = getattr(args, "_text_config", {}) or {}
        self.num_k_heads = text_cfg.get("linear_num_key_heads", 16)
        self.num_v_heads = text_cfg.get("linear_num_value_heads", 16)
        self.head_k_dim = text_cfg.get("linear_key_head_dim", 128)
        self.head_v_dim = text_cfg.get("linear_value_head_dim", 128)
        self.conv_kernel_size = text_cfg.get("linear_conv_kernel_dim", 4)
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim  # Q || K || V
        self.norm_eps = args.norm_eps

        # ============================================================
        # Host-side DeltaNet weights (kept on CPU; the host fallback
        # never moves these to device). All tensors are torch.bfloat16
        # except A_log/dt_bias which we promote to fp32 for stability.
        # ============================================================
        layer_prefix = f"layers.{layer_num}.linear_attn"

        def _get(name):
            key = f"{layer_prefix}.{name}"
            if key not in state_dict:
                raise KeyError(
                    f"LinearAttentionBlock(layer={layer_num}): missing weight '{key}'. "
                    f"Available linear_attn keys for this layer: "
                    f"{[k for k in state_dict.keys() if f'layers.{layer_num}.linear_attn' in k]}"
                )
            return state_dict[key]

        # in_proj_qkv: [conv_dim, hidden] → splits into Q (key_dim), K (key_dim), V (value_dim)
        self.W_in_qkv = _get("in_proj_qkv.weight").to(self.dtype_torch).clone()
        # in_proj_z: [value_dim, hidden] → gate
        self.W_in_z = _get("in_proj_z.weight").to(self.dtype_torch).clone()
        # in_proj_a, in_proj_b: [num_v_heads, hidden] each
        self.W_in_a = _get("in_proj_a.weight").to(self.dtype_torch).clone()
        self.W_in_b = _get("in_proj_b.weight").to(self.dtype_torch).clone()
        # conv1d.weight: [conv_dim, 1, K] depthwise
        self.W_conv = _get("conv1d.weight").to(self.dtype_torch).clone()
        # dt_bias, A_log: [num_v_heads]
        self.dt_bias = _get("dt_bias").float().clone()
        self.A_log = _get("A_log").float().clone()
        # out_proj: [hidden, value_dim]
        self.W_out = _get("out_proj.weight").to(self.dtype_torch).clone()
        # norm.weight: [head_v_dim]
        self.W_norm_gated = _get("norm.weight").to(self.dtype_torch).clone()

        # ============================================================
        # Per-user recurrent state. Lazily allocated on first forward
        # (we don't know B at construction). Persists across decode
        # steps; a fresh run should re-call reset_state().
        # ============================================================
        self._conv_state: Optional[torch.Tensor] = None   # [B, conv_dim, K-1]
        self._ssm_state: Optional[torch.Tensor] = None    # [B, HV, V, K]

        # ============================================================
        # Device modules: attention_norm, ffn_norm, MLP — identical to
        # TransformerBlock so weight-loading code paths still match.
        # ============================================================
        extra_rmsnorm_kwargs = {}
        if args.base_model_name in ("Qwen2.5-7B", "Qwen2.5-VL-7B"):
            extra_rmsnorm_kwargs["fp32_dest_acc_en"] = False

        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="attention_norm",
                is_distributed=self.args.is_distributed_norm,
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
                **extra_rmsnorm_kwargs,
            ),
            args,
            tt_ccl=self.tt_ccl,
            prefetcher=self.prefetcher,
            TG=args.is_galaxy,
            ag_config_key="ATTN_LN_AG_CONFIG",
        )
        self.ff_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="ffn_norm",
                is_distributed=self.args.is_distributed_norm,
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
                **extra_rmsnorm_kwargs,
            ),
            args,
            tt_ccl=self.tt_ccl,
            prefetcher=self.prefetcher,
            TG=args.is_galaxy,
            ag_config_key="FFN_LN_AG_CONFIG",
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=args.get_model_config(),
            prefetcher=prefetcher,
        )
        # Qwen3.5 doesn't use pre/post_feedforward_layernorm; keep None
        # so the TransformerBlock fast path semantics match.
        self.pre_ff_norm = None
        self.post_ff_norm = None

    # ----------------------------------------------------------------
    # State management
    # ----------------------------------------------------------------
    def reset_state(self):
        """Drop per-user conv/ssm state. Call between unrelated sequences."""
        self._conv_state = None
        self._ssm_state = None

    def _ensure_state(self, B: int, device: torch.device):
        if self._conv_state is None or self._conv_state.shape[0] != B:
            self._conv_state = torch.zeros(
                B, self.conv_dim, self.conv_kernel_size - 1,
                dtype=self.dtype_torch, device=device,
            )
        if self._ssm_state is None or self._ssm_state.shape[0] != B:
            self._ssm_state = torch.zeros(
                B, self.num_v_heads, self.head_v_dim, self.head_k_dim,
                dtype=torch.float32, device=device,
            )

    # ----------------------------------------------------------------
    # Host GatedDeltaNet kernel
    # ----------------------------------------------------------------
    def _host_delta_net_step(self, hidden: torch.Tensor) -> torch.Tensor:
        """One decode step of GatedDeltaNet on host CPU.

        Args:
            hidden: ``[B, hidden_size]`` post-attention_norm, replicated.

        Returns:
            ``[B, hidden_size]`` output of out_proj.
        """
        B = hidden.shape[0]
        self._ensure_state(B, hidden.device)

        # ----- (1) input projections --------------------------------------
        # qkv_proj: [B, conv_dim]
        qkv = torch.nn.functional.linear(hidden, self.W_in_qkv)
        z = torch.nn.functional.linear(hidden, self.W_in_z)
        a = torch.nn.functional.linear(hidden, self.W_in_a)
        b = torch.nn.functional.linear(hidden, self.W_in_b)

        # ----- (2) conv1d depthwise update (rolling) ----------------------
        # Slide the (K-1) history one position and append current qkv:
        #   conv_input[..., :K-1] = old_state
        #   conv_input[..., K-1] = qkv  (this step's value)
        # then convolve with W_conv (K taps per channel), bias=0, then SiLU.
        K = self.conv_kernel_size
        # build full input [B, conv_dim, K]
        conv_input = torch.cat([self._conv_state, qkv.unsqueeze(-1)], dim=-1)
        # depthwise conv: per channel, sum over K taps
        # W_conv shape: [conv_dim, 1, K] -> squeeze middle: [conv_dim, K]
        Wc = self.W_conv.squeeze(1)  # [conv_dim, K]
        qkv_post = (conv_input * Wc.unsqueeze(0)).sum(dim=-1)  # [B, conv_dim]
        qkv_post = torch.nn.functional.silu(qkv_post)
        # update rolling state: drop oldest, push current
        self._conv_state = conv_input[..., 1:].clone()

        # ----- (3) split qkv_post into Q, K, V ----------------------------
        q_flat = qkv_post[:, : self.key_dim]
        k_flat = qkv_post[:, self.key_dim : 2 * self.key_dim]
        v_flat = qkv_post[:, 2 * self.key_dim :]
        q = q_flat.reshape(B, self.num_k_heads, self.head_k_dim).float()
        k_t = k_flat.reshape(B, self.num_k_heads, self.head_k_dim).float()
        v_t = v_flat.reshape(B, self.num_v_heads, self.head_v_dim).float()

        # L2 norm Q and K (matches USE_QK_L2NORM_IN_KERNEL=True)
        q = q / (q.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
        k_t = k_t / (k_t.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
        scale = 1.0 / math.sqrt(self.head_k_dim)
        q = q * scale

        # If num_k_heads != num_v_heads, broadcast K over the group.
        # (For Qwen3.5-0.8B: num_k_heads = num_v_heads = 16, so no-op.)
        group = self.num_v_heads // self.num_k_heads
        if group != 1:
            k_t = k_t.repeat_interleave(group, dim=1)
            q = q.repeat_interleave(group, dim=1)

        # ----- (4) gating: g, beta ----------------------------------------
        a_f = a.float()
        b_f = b.float()
        x = a_f + self.dt_bias.unsqueeze(0)
        softplus = torch.where(
            x <= self.SOFTPLUS_THRESHOLD,
            torch.log1p(torch.exp(x)),
            x,
        )
        g = -torch.exp(self.A_log).unsqueeze(0) * softplus  # [B, HV]
        beta = torch.sigmoid(b_f)  # [B, HV]

        # ----- (5) state update (per head) --------------------------------
        # h: [B, HV, V, K]; here V = head_v_dim, K = head_k_dim
        h = self._ssm_state
        decay = torch.exp(g)  # [B, HV]
        h = h * decay[:, :, None, None]
        # k_view: [B, HV, K]
        # v_t : [B, HV, V]
        # delta = sum_K(h * k_view), shape [B, HV, V]
        delta = (h * k_t[:, :, None, :]).sum(dim=-1)
        v_corr = (v_t - delta) * beta[:, :, None]
        # outer(v_corr, k_view) shape [B, HV, V, K]
        h = h + v_corr[:, :, :, None] * k_t[:, :, None, :]
        # store updated state
        self._ssm_state = h
        # output: o[b, hv, v] = sum_K(h[b, hv, v, :] * q[b, hv, :])
        o = (h * q[:, :, None, :]).sum(dim=-1)  # [B, HV, V]

        # ----- (6) RMSNormGated then out_proj -----------------------------
        z_view = z.reshape(B, self.num_v_heads, self.head_v_dim)
        o = _rms_norm_gated(
            o.to(self.dtype_torch),
            z_view,
            self.W_norm_gated,
            self.norm_eps,
        )
        o_flat = o.reshape(B, self.value_dim)
        out = torch.nn.functional.linear(o_flat, self.W_out)
        return out

    # ----------------------------------------------------------------
    # ttnn ↔ host bridge
    # ----------------------------------------------------------------
    def _to_host_replicated(self, x_tt: ttnn.Tensor) -> torch.Tensor:
        """Materialize a replicated TT tensor onto host CPU as torch.

        After ``attention_norm`` with all_gather, every device has the same
        full hidden replica (modulo numerical noise from the gather), so
        device 0's view IS the full hidden activation. We trim leading
        singleton dims so callers see ``[B, hidden]``.
        """
        device_tensors = ttnn.get_device_tensors(x_tt)
        host = ttnn.to_torch(device_tensors[0])  # typically [1, 1, B, H]
        # Reduce to 2D [B, H]; squeeze only true singletons so we don't
        # eat a batch=1 axis.
        if host.dim() == 4 and host.shape[0] == 1 and host.shape[1] == 1:
            host = host.reshape(host.shape[2], host.shape[3])
        elif host.dim() == 3 and host.shape[0] == 1:
            host = host.reshape(host.shape[1], host.shape[2])
        return host

    def _from_host_to_device(self, x_host: torch.Tensor, ref_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Push host output to device, sharded across the hidden dim.

        The residual ``x`` is width-sharded across the mesh (each device
        holds ``hidden/num_devices``). To make the downstream residual-add
        work, our ``attn_out`` must end up sharded the same way.

        Approach: shard the FULL hidden output across mesh dim=-1 using
        ``ShardTensorToMesh``; ttnn auto-splits the host tensor between
        devices. Land in DRAM_INTERLEAVED; the caller does
        ``ttnn.to_memory_config(skip_mem_cfg)`` to push it into L1 if
        needed (this mirrors TransformerBlock which also does a
        post-attention to_memory_config).

        x_host: ``[B, hidden]`` (FULL hidden on host).
        Returns a 4D [1, 1, B, hidden/num_devices] sharded device tensor.
        """
        if x_host.dim() == 2:
            x_host = x_host.unsqueeze(0).unsqueeze(0)
        elif x_host.dim() == 3:
            x_host = x_host.unsqueeze(0)
        # Decide between Replicate and Shard based on the residual's
        # per-device hidden dim. If the residual is full-hidden replicated
        # (single-device or unusual layouts), replicate too.
        num_devs = self.args.num_devices
        if num_devs > 1 and x_host.shape[-1] % num_devs == 0:
            mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=-1)
        else:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        tt = ttnn.from_torch(
            x_host.to(torch.bfloat16),
            device=self.mesh_device,
            dtype=ref_tt.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        return tt

    # ----------------------------------------------------------------
    # Block forward
    # ----------------------------------------------------------------
    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ) -> ttnn.Tensor:
        """Mirror TransformerBlock.forward but route the attention sub-block
        through the host fallback. MLP + residual + norms still run on device.
        """
        TG = self.args.is_galaxy
        residual = x
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"linear-attn input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        # (1) attention_norm on device
        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        attn_in = self.attention_norm(x, mode, norm_config=attn_norm_config)

        # (2) Host fallback: gather, run DeltaNet, push back
        host_hidden = self._to_host_replicated(attn_in)
        # host_hidden shape: [S, hidden] (decode S=batch_size; prefill S=seq_len)
        # For decode we expect a 1-step per-user batch [B, hidden]; reshape if needed
        if host_hidden.dim() == 1:
            host_hidden = host_hidden.unsqueeze(0)
        host_out = self._host_delta_net_step(host_hidden)
        attn_out = self._from_host_to_device(host_out, ref_tt=x)
        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)

        # (3) Residual add, ffn_norm, MLP, residual add — same as TransformerBlock
        hidden_states = ttnn.add(
            residual, attn_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None
        )
        residual = hidden_states
        if mode == "prefill":
            try:
                x.deallocate(True)
            except Exception:
                pass

        ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
        hidden_states = self.ff_norm(hidden_states, mode, norm_config=ff_norm_config)
        ttnn.deallocate(attn_out)

        if TG and mode == "decode":
            hidden_states = ttnn.to_memory_config(
                hidden_states, memory_config=self.args.get_mlp_act_mem_config(mode)
            )
        hidden_states = self.feed_forward.forward(hidden_states, mode)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=skip_mem_cfg,
            dtype=ttnn.bfloat16,
        )
        return out
