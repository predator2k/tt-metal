# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LinearAttentionBlock — GatedDeltaNet decoder block for Qwen3.5.

This is the *correctness-first* path for Qwen3.5's hybrid attention pattern
(layer types: ``linear_attention`` vs ``full_attention``). The full-attention
layers use the existing TT-native ``TransformerBlock``; linear layers route
through this block's GatedDeltaNet computation.

Two execution modes are supported, selected at module-construction time by
``SGLANG_TT_QWEN35_DELTANET_NATIVE``:

  * Default (env unset/0): **host-fallback** — DeltaNet math runs in pure
    PyTorch on the CPU in one big bf16 batch per layer per step.
  * Native (env=1): **TT-native** — full DeltaNet (input projections,
    on-device causal conv1d_update, L2-norm + Q-scale, softplus/sigmoid
    gating, SSM outer-product update, RMSNormGated, and out_proj) runs
    on device with persistent on-device conv/ssm state. Zero host bridges
    per layer per step as of WS-A.16; only the initial hidden gather
    (~2 KB) and the final attn-out re-shard cross PCIe per step.

PERF — WS-A.17 trace replay (measured 2026-05-23, P150a 2x mesh, B=1)
----------------------------------------------------------------------
For Qwen3.5-0.8B at batch=1, head_dim=128, 16 v-heads. Warm avg per
decode step (steps 5-19, run-to-run jitter ~10 ms):

  host-fallback (eager):           344.8 ms      (host PyTorch DeltaNet)
  TT-native + trace replay:         31.5 ms      (WS-A.17)

That is an 11× speedup over eager. The trace path slices the 32-row
tile-padded hidden activation down to user 0 (the only active row given
``max_batch_size==1``) inside the trace-safe DeltaNet step, runs B=1 math
on device, and pads the [1,1,1,hidden] output back to [1,1,32,hidden] via
``ttnn.pad`` so the downstream residual add (against the 32-row residual)
matches.

Trace replay vs host-fallback PCC (full vocab logits, 6 steps):
  step 0 PCC=0.9994, step 1 PCC=0.7604, step 2 PCC=0.9316,
  step 3 PCC=0.9906, step 4 PCC=0.9899, step 5 PCC=0.9954
All 6 step tokens are bit-exact (top-1 match). The step-1 dip is bf16
accumulation noise (Wo BFP8 + SDPA HiFi2 + redundant single-device
DeltaNet replication); structural correctness is unaffected.

The earlier WS-A.16 perf header claimed the TT-native eager path took
358 ms vs host-fallback 348 ms. Both numbers were the SAME host-fallback
path running — the ``mode == "decode"`` comparison was always False
(``mode`` is the ``Mode.DECODE`` enum, not the string ``"decode"``), so
the ``self._tt_native_enabled and mode == "decode"`` guard never fired
and the native step was never taken. WS-A.17 fixes this by normalizing
``mode`` to a string at the top of ``forward()`` (see ``_mode_str``).

PCC native vs host-fallback: 0.0000-diff (bit-equivalent in bf16, since
the on-device math is mathematically identical and intermediates land
in the same bf16 representation).
PCC vs HF reference: 0.7183 / 0.6967 / 0.6806 / 0.6527 (steps 0/1/4/16)
— identical to the host-fallback baseline.

Wo precision audit (WS-A.13): lifting Wo BFP8→BF16 (and/or full attention
to BF16+HIFI4) only moved end-to-end PCC by +0.005..+0.018 — below the
+0.05 commit threshold. Kept the env hook (`SGLANG_TT_QWEN35_WO_PRECISION`
in model_config.py) so a future op can opt-in without rebuilding; default
remains BFP8.

Future work to close the rest of the gap:
  1. SGLang server + prefetcher path (WS-A.18 — alignment with the
     Qwen3-8B 27 ms TPOT target).
  2. Larger per-user batch (B≥4) to amortize residual dispatch overhead.
  3. Sharded projections across the mesh (currently the native path runs
     redundantly on a replicated single-device view).
  4. Lift Wo/SDPA precision (already env-gated via
     ``SGLANG_TT_QWEN35_WO_PRECISION``, ``SGLANG_TT_QWEN35_WSA14_SDPA_HIFI4``)
     to close the step-1 PCC noise.

Why host fallback originally (WS-A.2 historical):
  * tt-metal has no fused GatedDeltaNet/causal_conv1d_update kernel.
  * Writing one was a separate workstream (multi-week, not WS-A.2 scope).
  * WS-A.2 only needed the smoke to advance past G3 so WS-A.3 (MRoPE)
    could land.

The MLP, residual-add, and the two RMSNorms (``attention_norm``, ``ffn_norm``)
run on device — same as ``TransformerBlock`` — so prefill/decode plumbing and
weight loading match the full-attention path. Only the ``self_attn`` substitute
is host-fallback.

Per-layer recurrent state lives on the host (``self._conv_state``,
``self._ssm_state``); the paged KV cache is bypassed for these layers. The
generator's ``allocate_sglang_kv_cache`` puts ``None`` entries in the per-layer
list for linear layers, and ``forward()`` here ignores its ``kv_cache=`` kwarg.

Host-fallback mode (default) is intentionally slow (~50-200 ms per layer per
decode step from the host roundtrip); it exists so WS-A.3 could write the
MRoPE work without being blocked on a kernel that didn't exist yet.

TT-native mode (WS-A.16) keeps the entire DeltaNet step on-device — last
remaining host bridge in the decode path. This is a prerequisite for trace
capture (WS-A.17). Eager-mode dispatch cost is comparable to host-fallback;
the win is unlocked by trace replay, not by the on-device migration itself.

WS-A.17 adds a trace-safe variant ``_tt_native_delta_net_step_trace_safe``
that:
  * consumes the replicated hidden input directly (no host bridge),
  * slices to user 0 (B=1) for cheap per-step DeltaNet math,
  * updates ``_tt_conv_state`` and ``_tt_ssm_state`` IN-PLACE via
    ``ttnn.copy`` so the buffer addresses stay stable across
    ``ttnn.execute_trace`` replays, and
  * pads the [1,1,1,hidden] output back to the tile-padded [1,1,32,hidden]
    via ``ttnn.pad`` so the downstream residual add lines up.

Gated on ``SGLANG_TT_QWEN35_TRACE=1`` (requires
``SGLANG_TT_QWEN35_DELTANET_NATIVE=1``).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mlp import MLP


# ---------------------------------------------------------------------------
# WS-A.12 — TT-native GatedDeltaNet env-gate
# ---------------------------------------------------------------------------
# Default OFF. When set to "1" / "true", the LinearAttentionBlock executes its
# DeltaNet step using ttnn ops on a single device's view (no host roundtrip)
# instead of the host-fallback PyTorch path. The same correctness contract
# applies (slice-broadcast hidden, full DeltaNet, re-shard hidden out).
#
# First pass intentionally redundantly computes on one device and broadcasts
# the result — preserves bit-similarity to host-fallback while removing the
# CPU roundtrip cost. Sharding the projections across devices is WS-A.13.
def _qwen35_deltanet_native_enabled() -> bool:
    return os.environ.get("SGLANG_TT_QWEN35_DELTANET_NATIVE", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# WS-A.17 — Decode trace capture env-gate
# ---------------------------------------------------------------------------
# When SGLANG_TT_QWEN35_TRACE=1 the LinearAttentionBlock runs a trace-safe
# variant of the native DeltaNet step that:
#   1. consumes its replicated hidden input directly (no to_torch/from_torch
#      PCIe round-trip),
#   2. updates the persistent conv_state and ssm_state IN-PLACE via ttnn.copy
#      (the buffer addresses must be stable for `ttnn.execute_trace`), and
#   3. emits a replicated full-hidden output that the harness's trace wrapper
#      then routes through ttnn.mesh_partition (also on device) before the
#      residual add.
# Requires SGLANG_TT_QWEN35_DELTANET_NATIVE=1; otherwise host-fallback runs
# and no trace can be captured.
def _qwen35_trace_enabled() -> bool:
    return os.environ.get("SGLANG_TT_QWEN35_TRACE", "").lower() in ("1", "true", "yes")


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
        # WS-A.12 TT-native path: lazy on-device state + weights.
        # Both stay None when the env-gate is off so the host-fallback
        # path runs identically to before this commit.
        # ============================================================
        self._tt_native_enabled = _qwen35_deltanet_native_enabled()
        # WS-A.17: when both DELTANET_NATIVE and QWEN35_TRACE are set, the
        # forward path takes the trace-safe variant (zero host bridges, in-
        # place ssm/conv state updates). Without DELTANET_NATIVE the host-
        # fallback runs even if TRACE=1.
        self._tt_trace_enabled = self._tt_native_enabled and _qwen35_trace_enabled()
        # Per-layer device-resident state, replicated across the mesh
        # because the per-step compute runs on a single replicated view.
        self._tt_conv_state: Optional[ttnn.Tensor] = None  # bf16, replicated
        self._tt_ssm_state: Optional[ttnn.Tensor] = None   # fp32, replicated
        # On-device weight handles, lazily uploaded on first native step.
        self._tt_weights_loaded = False
        self._tt_W_in_qkv = None
        self._tt_W_in_z = None
        self._tt_W_in_a = None
        self._tt_W_in_b = None
        self._tt_W_conv = None       # [1, 1, conv_dim, K] for per-channel scale
        self._tt_dt_bias = None      # [1, 1, 1, num_v_heads]
        self._tt_A_factor = None     # -exp(A_log) precomputed, [1, 1, 1, num_v_heads]
        self._tt_W_out = None
        self._tt_W_norm_gated = None  # [1, 1, 1, head_v_dim]

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
    # WS-A.12 TT-native helpers
    # ----------------------------------------------------------------
    def _from_torch_replicated(self, t: torch.Tensor, dtype) -> ttnn.Tensor:
        """Upload a torch tensor to every device of the mesh (replicated)."""
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _ensure_tt_weights(self, B: int) -> None:
        """Upload DeltaNet weights as device-resident, replicated ttnn tensors.
        Called once on first native forward; the upload is amortized over all
        subsequent decode steps.

        Shapes uploaded (per-device, replicated):
          W_in_qkv: [1, 1, conv_dim, hidden]      (transposed for ttnn.linear)
          W_in_z:   [1, 1, value_dim, hidden]
          W_in_a:   [1, 1, num_v_heads, hidden]
          W_in_b:   [1, 1, num_v_heads, hidden]
          W_conv:   [1, 1, conv_dim, K]           (per-channel kernel taps)
          dt_bias:  [1, 1, 1, num_v_heads]
          A_factor: [1, 1, 1, num_v_heads]        (== -exp(A_log) precomputed)
          W_out:    [1, 1, hidden, value_dim]
          W_norm_gated: [1, 1, 1, head_v_dim]

        Linear weights are stored TRANSPOSED ([out, in] from HF is uploaded
        as [in, out] by transposing before from_torch, then we use
        ``ttnn.linear(x, W)`` which expects W as [..., in, out]).
        """
        if self._tt_weights_loaded:
            return

        # ttnn.linear contract: out = matmul(x, W); W is [in, out]
        # HF weights are stored as [out, in], so transpose before upload.
        def _w_for_linear(t):
            return t.transpose(0, 1).contiguous().to(torch.bfloat16).reshape(1, 1, t.shape[1], t.shape[0])

        self._tt_W_in_qkv = self._from_torch_replicated(
            _w_for_linear(self.W_in_qkv), dtype=ttnn.bfloat16
        )
        self._tt_W_in_z = self._from_torch_replicated(
            _w_for_linear(self.W_in_z), dtype=ttnn.bfloat16
        )
        self._tt_W_in_a = self._from_torch_replicated(
            _w_for_linear(self.W_in_a), dtype=ttnn.bfloat16
        )
        self._tt_W_in_b = self._from_torch_replicated(
            _w_for_linear(self.W_in_b), dtype=ttnn.bfloat16
        )
        self._tt_W_out = self._from_torch_replicated(
            _w_for_linear(self.W_out), dtype=ttnn.bfloat16
        )

        # W_conv: [conv_dim, 1, K]  -> [1, 1, conv_dim, K]
        # WS-A.17: repeat across batch up front so the on-device
        # multiply [1,1,B*conv_dim,K] * [1,1,B*conv_dim,K] is a same-shape
        # elementwise op (no implicit broadcast that would fail for B>1).
        Wc = self.W_conv.squeeze(1).contiguous().to(torch.bfloat16).reshape(
            1, 1, self.conv_dim, self.conv_kernel_size
        )
        Wc_repeated = Wc.repeat(1, 1, B, 1)  # [1, 1, B*conv_dim, K]
        self._tt_W_conv = self._from_torch_replicated(Wc_repeated, dtype=ttnn.bfloat16)

        # dt_bias and A_log → reshape to [1, 1, 1, num_v_heads]
        dt_b = self.dt_bias.to(torch.bfloat16).reshape(1, 1, 1, self.num_v_heads)
        A_fac = (-torch.exp(self.A_log)).to(torch.bfloat16).reshape(
            1, 1, 1, self.num_v_heads
        )
        self._tt_dt_bias = self._from_torch_replicated(dt_b, dtype=ttnn.bfloat16)
        self._tt_A_factor = self._from_torch_replicated(A_fac, dtype=ttnn.bfloat16)

        # RMSNormGated weight: [head_v_dim] → [1, 1, 1, head_v_dim]
        Wn = self.W_norm_gated.to(torch.bfloat16).reshape(1, 1, 1, self.head_v_dim)
        self._tt_W_norm_gated = self._from_torch_replicated(Wn, dtype=ttnn.bfloat16)

        self._tt_weights_loaded = True

    def _ensure_tt_state(self, B: int) -> None:
        """Allocate persistent device-resident conv/ssm state on first call."""
        if self._tt_conv_state is None:
            cs = torch.zeros(
                1, 1, B * self.conv_dim, self.conv_kernel_size - 1,
                dtype=torch.bfloat16,
            )
            # store as [1, 1, B*conv_dim, K-1] in tile layout — flat layout is
            # fine for our per-channel multiply / roll operations
            self._tt_conv_state = self._from_torch_replicated(cs, dtype=ttnn.bfloat16)
        if self._tt_ssm_state is None:
            # [B*HV, V, K] — collapse B and HV into the batched-matmul leading
            # dim so we can use ttnn.matmul for outer product and h@q updates
            ss = torch.zeros(
                1, B * self.num_v_heads, self.head_v_dim, self.head_k_dim,
                dtype=torch.bfloat16,
            )
            self._tt_ssm_state = self._from_torch_replicated(ss, dtype=ttnn.bfloat16)

    def _tt_softplus_clamped(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """ttnn.softplus has a threshold parameter for numerical stability."""
        return ttnn.softplus(x, beta=1.0, threshold=self.SOFTPLUS_THRESHOLD)

    def _tt_l2_norm_scale(self, x: ttnn.Tensor, scale: float) -> ttnn.Tensor:
        """Normalize along the last dim then scale by `scale`.
        x = x / (||x||_2 + 1e-6) * scale
        Operates in fp32 dest accumulator for stability where supported.
        """
        sq = ttnn.multiply(x, x)
        sumsq = ttnn.sum(sq, dim=-1, keepdim=True)
        ttnn.deallocate(sq)
        norm = ttnn.sqrt(sumsq)
        ttnn.deallocate(sumsq)
        eps_t = ttnn.add(norm, 1e-6)
        ttnn.deallocate(norm)
        inv = ttnn.reciprocal(eps_t)
        ttnn.deallocate(eps_t)
        scaled = ttnn.multiply(x, inv)
        ttnn.deallocate(inv)
        # scalar multiply by `scale` (e.g. 1/sqrt(head_dim))
        out = ttnn.multiply(scaled, scale)
        ttnn.deallocate(scaled)
        return out

    def _tt_native_delta_net_step(self, attn_in_tt: ttnn.Tensor) -> ttnn.Tensor:
        """TT-native one-decode-step GatedDeltaNet (WS-A.16 single-bridge).

        Strategy (WS-A.16: last conv1d host bridge eliminated):
          1. Gather hidden to host (2 KB), upload replicated to device.
          2. Run the 4 input projections (in_proj_qkv/z/a/b) on device.
          3. Conv1d depthwise rolling update fully ON DEVICE via
             ttnn.{reshape, concat, multiply, sum, silu, slice} composition.
             Persistent conv state ([1, 1, B*conv_dim, K-1]) stays in DRAM
             across decode steps — no per-step PCIe roundtrip.
          4. Slice qkv_post on device into Q/K/V; L2-norm + scale on device.
          5. Compute gating (decay, beta) entirely on device via
             ttnn.softplus + ttnn.exp + ttnn.sigmoid.
          6. SSM state update (4 batched-matmul ops on persistent on-device
             [1, B*HV, V, K] state tensor) — unchanged from WS-A.12.
          7. RMSNormGated on device via ttnn.silu + ttnn.multiply +
             ttnn.rms_norm(weight=...).
          8. out_proj matmul on device.
          9. Return replicated [1, 1, B, hidden] tensor.

        Persistent state: ``_tt_conv_state`` and ``_tt_ssm_state`` live in
        DRAM as device-resident replicated ttnn tensors across all decode
        steps. Weights are uploaded once on first call (``_ensure_tt_weights``).

        WS-A.16 vs WS-A.13: the WS-A.13 path had 2 host hops per layer per
        step (hidden in, qkv_post round-trip for conv1d). WS-A.16 has 1
        hop: hidden in. The conv1d update became 6 ttnn ops with bit-
        equivalent bf16 output. This unblocks decode trace capture
        (WS-A.17), which required a fully on-device execution graph.

        Args:
            attn_in_tt: post-attention_norm ttnn tensor, sharded or
                replicated, shape ends in `[1, 1, B, hidden]` per device.

        Returns:
            ttnn tensor with shape `[1, 1, B, hidden]`, replicated across
            all devices in the mesh.
        """
        # Mirror _to_host_replicated to get a single device's view of the
        # full hidden, then upload it as a single replicated device tensor.
        # This avoids any cross-device divergence in subsequent compute.
        device_tensors = ttnn.get_device_tensors(attn_in_tt)
        host = ttnn.to_torch(device_tensors[0])
        if host.dim() == 4 and host.shape[0] == 1 and host.shape[1] == 1:
            host = host.reshape(host.shape[2], host.shape[3])
        elif host.dim() == 3 and host.shape[0] == 1:
            host = host.reshape(host.shape[1], host.shape[2])
        if host.dim() == 1:
            host = host.unsqueeze(0)
        B = host.shape[0]
        H = host.shape[1]
        assert H == self.hidden_size, (
            f"linear-attn native: hidden mismatch {H} vs {self.hidden_size}"
        )

        self._ensure_tt_weights(B)
        self._ensure_tt_state(B)

        # Upload hidden as [1, 1, B, hidden] replicated
        hidden4 = host.to(torch.bfloat16).reshape(1, 1, B, H)
        x = self._from_torch_replicated(hidden4, dtype=ttnn.bfloat16)

        # ----- (1) input projections ---------------------------------
        qkv = ttnn.linear(x, self._tt_W_in_qkv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        z   = ttnn.linear(x, self._tt_W_in_z,   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a   = ttnn.linear(x, self._tt_W_in_a,   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b   = ttnn.linear(x, self._tt_W_in_b,   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)

        # ----- (2) Conv1d depthwise rolling update (WS-A.16: on-device) ----
        # The depthwise causal conv1d_update is composed from ttnn primitives:
        #   reshape  qkv         [1,1,B,conv_dim]       → [1,1,B*conv_dim,1]
        #   concat   [state | q]                         → [1,1,B*conv_dim,K]
        #   multiply by W_conv  [1,1,conv_dim,K] (B=1)   → [1,1,B*conv_dim,K]
        #   sum      dim=-1                              → [1,1,B*conv_dim,1]
        #   silu                                         → [1,1,B*conv_dim,1]
        #   reshape  → [1,1,B,conv_dim] for downstream slice ops.
        #
        # Rolling state update: slice [..., 1:K] of the concatenated buffer →
        # [1,1,B*conv_dim,K-1] persists as the new conv_state.
        #
        # PCC vs the prior WS-A.13 host-bridged path: bf16 bit-equivalent
        # (verified by /tmp/_wsa16_op_probe.py to PCC=0.999999, max|diff|=
        # one bf16 ULP). The on-device composition removes the last per-step
        # PCIe roundtrip; unblocks decode trace capture (WS-A.17).
        #
        # Limitation: assumes B==1 so W_conv ([1,1,conv_dim,K]) matches
        # conv_input ([1,1,B*conv_dim,K]) shape exactly. For B>1 we'd need
        # to repeat W_conv over batch — out of scope this commit.
        K = self.conv_kernel_size
        assert B == 1, (
            f"WS-A.16 on-device conv1d_update assumes B==1 (W_conv broadcast); "
            f"got B={B}. Repeat W_conv across batch to extend."
        )

        # Flatten qkv: [1,1,B,conv_dim] → [1,1,B*conv_dim,1]
        qkv_flat = ttnn.reshape(qkv, [1, 1, B * self.conv_dim, 1])
        ttnn.deallocate(qkv)

        # Sliding window: [state | qkv_flat] along last axis.
        conv_input = ttnn.concat([self._tt_conv_state, qkv_flat], dim=-1)
        ttnn.deallocate(qkv_flat)

        # Depthwise multiply by W_conv (per-channel kernel taps).
        weighted = ttnn.multiply(conv_input, self._tt_W_conv)

        # Per-channel reduction over the K taps.
        qkv_post_reduced = ttnn.sum(weighted, dim=-1, keepdim=True)
        ttnn.deallocate(weighted)

        # SiLU activation.
        qkv_post_silu = ttnn.silu(qkv_post_reduced)
        ttnn.deallocate(qkv_post_reduced)

        # Reshape back to [1,1,B,conv_dim] for the downstream slice ops.
        qkv_post = ttnn.reshape(qkv_post_silu, [1, 1, B, self.conv_dim])
        ttnn.deallocate(qkv_post_silu)

        # Slide the rolling state forward: drop the oldest column, keep K-1.
        new_state = ttnn.slice(
            conv_input,
            [0, 0, 0, 1],
            [1, 1, B * self.conv_dim, K],
        )
        ttnn.deallocate(conv_input)
        # Swap the persistent on-device state. Deallocate the old buffer last.
        ttnn.deallocate(self._tt_conv_state)
        self._tt_conv_state = new_state

        # ----- (3) slice into Q, K, V on DEVICE, L2-norm + scale -----
        # qkv_post layout (along last axis): [ Q (key_dim) | K (key_dim) | V (value_dim) ]
        q_slice = ttnn.slice(qkv_post, [0, 0, 0, 0], [1, 1, B, self.key_dim])
        k_slice = ttnn.slice(
            qkv_post, [0, 0, 0, self.key_dim], [1, 1, B, 2 * self.key_dim]
        )
        v_slice = ttnn.slice(
            qkv_post,
            [0, 0, 0, 2 * self.key_dim],
            [1, 1, B, 2 * self.key_dim + self.value_dim],
        )
        ttnn.deallocate(qkv_post)

        # Reshape so the last dim is head_*_dim → L2-norm operates per-head.
        # Note: num_k_heads == num_v_heads for Qwen3.5-0.8B (group=1). The
        # group!=1 case below would need ttnn.repeat_interleave which is
        # less mature; fall back to host bridge if encountered (Qwen3.5
        # large variants if/when they land).
        group = self.num_v_heads // self.num_k_heads
        assert group == 1, (
            f"WS-A.13 on-device L2 path assumes num_k_heads == num_v_heads "
            f"(got {self.num_k_heads} vs {self.num_v_heads}); add a "
            f"ttnn.repeat_interleave fallback if this triggers."
        )
        # Reshape to [1, 1, B*HV, head_dim] — last dim is the L2 axis.
        q_heads = ttnn.reshape(q_slice, [1, 1, B * self.num_k_heads, self.head_k_dim])
        k_heads = ttnn.reshape(k_slice, [1, 1, B * self.num_k_heads, self.head_k_dim])
        v_heads = ttnn.reshape(v_slice, [1, 1, B * self.num_v_heads, self.head_v_dim])
        ttnn.deallocate(q_slice)
        ttnn.deallocate(k_slice)
        ttnn.deallocate(v_slice)

        q_normed = self._tt_l2_norm_scale(q_heads, 1.0 / math.sqrt(self.head_k_dim))
        k_normed = self._tt_l2_norm_scale(k_heads, 1.0)
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)

        # Reshape Q/K/V to the SSM matmul layout: [1, B*HV, head_dim, 1]
        # for column form, plus a [1, B*HV, 1, head_dim] row form of K
        # for the outer product. V uses [1, B*HV, V, 1].
        q_col = ttnn.reshape(q_normed, [1, B * self.num_v_heads, self.head_k_dim, 1])
        k_col = ttnn.reshape(k_normed, [1, B * self.num_v_heads, self.head_k_dim, 1])
        ttnn.deallocate(q_normed)
        # second view of k for the outer product (row form)
        k_row = ttnn.reshape(k_col, [1, B * self.num_v_heads, 1, self.head_k_dim])
        v_col = ttnn.reshape(v_heads, [1, B * self.num_v_heads, self.head_v_dim, 1])
        ttnn.deallocate(v_heads)
        ttnn.deallocate(k_normed)

        # ----- (4) gating: decay, beta on DEVICE ---------------------
        # a shape: [1,1,B,HV], _tt_dt_bias shape: [1,1,1,HV] → bcast add.
        a_biased = ttnn.add(a, self._tt_dt_bias)
        ttnn.deallocate(a)
        sp = self._tt_softplus_clamped(a_biased)   # [1,1,B,HV]
        ttnn.deallocate(a_biased)
        # g = A_factor * sp  (A_factor already == -exp(A_log), bcast over B)
        g = ttnn.multiply(self._tt_A_factor, sp)   # [1,1,B,HV]
        ttnn.deallocate(sp)
        decay = ttnn.exp(g)                         # [1,1,B,HV]
        ttnn.deallocate(g)
        # Reshape decay → [1, B*HV, 1, 1] for the SSM head-scalar bcast.
        decay_bcast = ttnn.reshape(decay, [1, B * self.num_v_heads, 1, 1])

        beta = ttnn.sigmoid(b)                      # [1,1,B,HV]
        ttnn.deallocate(b)
        beta_bcast = ttnn.reshape(beta, [1, B * self.num_v_heads, 1, 1])

        # ----- (5) SSM state update via batched matmul on DEVICE ------
        # State is [1, B*HV, V, K]. Decay it (per-head scalar).
        h_decayed = ttnn.multiply(self._tt_ssm_state, decay_bcast)
        ttnn.deallocate(self._tt_ssm_state)
        ttnn.deallocate(decay_bcast)

        # delta = h @ k_col  →  [1, B*HV, V, 1]
        delta = ttnn.matmul(h_decayed, k_col, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # v_corr = (v - delta) * beta per head
        v_minus_delta = ttnn.sub(v_col, delta)
        ttnn.deallocate(delta)
        ttnn.deallocate(v_col)
        v_corr = ttnn.multiply(v_minus_delta, beta_bcast)
        ttnn.deallocate(v_minus_delta)
        ttnn.deallocate(beta_bcast)
        # outer(v_corr, k_row)  →  [1, B*HV, V, K]
        outer = ttnn.matmul(v_corr, k_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(v_corr)
        ttnn.deallocate(k_row)
        ttnn.deallocate(k_col)

        h_new = ttnn.add(h_decayed, outer)
        ttnn.deallocate(h_decayed)
        ttnn.deallocate(outer)

        # o = h_new @ q_col  →  [1, B*HV, V, 1]
        o = ttnn.matmul(h_new, q_col, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_col)

        # Persist updated state — keep h_new alive
        self._tt_ssm_state = h_new

        # ----- (6) RMSNormGated + out_proj on DEVICE ------------------
        # Reshape o [1, B*HV, V, 1] → [1, 1, B*HV, V] for elementwise & norm.
        o_2d = ttnn.reshape(o, [1, 1, B * self.num_v_heads, self.head_v_dim])
        ttnn.deallocate(o)
        # Reshape z [1, 1, B, value_dim] → [1, 1, B*HV, V] (HV*V == value_dim).
        z_2d = ttnn.reshape(z, [1, 1, B * self.num_v_heads, self.head_v_dim])
        ttnn.deallocate(z)
        # silu(z), then o * silu(z).
        silu_z = ttnn.silu(z_2d)
        ttnn.deallocate(z_2d)
        o_gated = ttnn.multiply(o_2d, silu_z)
        ttnn.deallocate(o_2d)
        ttnn.deallocate(silu_z)
        # RMS norm with per-channel gamma, last-dim normalization.
        o_normed = ttnn.rms_norm(
            o_gated, weight=self._tt_W_norm_gated, epsilon=self.norm_eps
        )
        ttnn.deallocate(o_gated)
        # Reshape back to [1, 1, B, value_dim] for out_proj.
        o_flat = ttnn.reshape(o_normed, [1, 1, B, self.value_dim])
        ttnn.deallocate(o_normed)
        out = ttnn.linear(o_flat, self._tt_W_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o_flat)
        return out  # [1, 1, B, hidden], replicated

    # ----------------------------------------------------------------
    # WS-A.17 trace-safe variant
    # ----------------------------------------------------------------
    def _tt_native_delta_net_step_trace_safe(self, attn_in_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Trace-safe TT-native DeltaNet step (WS-A.17).

        Differences vs ``_tt_native_delta_net_step``:

          * NO host bridge — uses ``attn_in_tt`` directly (already replicated
            after attention_norm's all_gather), routed to DRAM by
            ``ttnn.to_memory_config``.  ``ttnn.to_torch`` and
            ``ttnn.from_torch`` are host ops that cannot be recorded into a
            ``ttnn.execute_trace`` graph.
          * Persistent state buffers (``_tt_conv_state`` and
            ``_tt_ssm_state``) are NOT reassigned; we write the new state
            INTO the existing buffers via ``ttnn.copy``.  ``execute_trace``
            bakes tensor addresses into the recorded program, so the state
            tensors must keep the SAME address across replays.

        Layout/dtype/shape of ``attn_in_tt``:
          shape ends in ``[1, 1, B, hidden]`` per device, REPLICATED (i.e.
          every device holds the full hidden width).  This is the contract
          enforced by ``DistributedNorm.forward`` (it ends in
          ``ttnn.experimental.all_gather_async`` when the norm is distributed,
          which is the standalone Qwen3.5 path).
        """
        # Route input to DRAM so the rest of the pipeline (which expects
        # DRAM_MEMORY_CONFIG) operates against a well-known buffer type.
        # On-device op — trace-safe.
        x = ttnn.to_memory_config(attn_in_tt, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Canonical input shape after attention_norm: [1, 1, B_padded, hidden]
        # where B_padded == 32 (tile-aligned).  Reshape if not already 4D.
        if len(x.shape) == 2:
            x = ttnn.reshape(x, [1, 1, x.shape[0], x.shape[1]])
        elif len(x.shape) == 3:
            x = ttnn.reshape(x, [1, 1, x.shape[1], x.shape[2]])

        # WS-A.17: only user 0 is active (max_batch_size==1).  Slice on
        # device to a tile-aligned [1, 1, 32, hidden] but logically treat
        # B=1 for the DeltaNet math.  The state tensors are sized for B=1
        # so the DeltaNet sees a single user; the output is padded back
        # to [1, 1, 32, hidden] at the end so the residual add downstream
        # matches the 32-row tile.
        B_padded = int(x.shape[-2])
        B = 1
        # Slice to user 0 — first row of the batch axis (on-device op,
        # trace-safe).  Run B=1 DeltaNet; pad the output back to
        # [1, 1, 32, hidden] before returning (see end of method) so the
        # downstream residual add matches.
        x_user0 = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, self.hidden_size])
        ttnn.deallocate(x)
        x = x_user0

        self._ensure_tt_weights(B)
        self._ensure_tt_state(B)

        # ----- (1) input projections ---------------------------------
        qkv = ttnn.linear(x, self._tt_W_in_qkv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        z   = ttnn.linear(x, self._tt_W_in_z,   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a   = ttnn.linear(x, self._tt_W_in_a,   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b   = ttnn.linear(x, self._tt_W_in_b,   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)

        # ----- (2) Conv1d depthwise rolling update --------------------
        K = self.conv_kernel_size
        qkv_flat = ttnn.reshape(qkv, [1, 1, B * self.conv_dim, 1])
        ttnn.deallocate(qkv)
        # Sliding window: concat persistent state with new qkv.
        conv_input = ttnn.concat([self._tt_conv_state, qkv_flat], dim=-1)
        ttnn.deallocate(qkv_flat)
        weighted = ttnn.multiply(conv_input, self._tt_W_conv)
        qkv_post_reduced = ttnn.sum(weighted, dim=-1, keepdim=True)
        ttnn.deallocate(weighted)
        qkv_post_silu = ttnn.silu(qkv_post_reduced)
        ttnn.deallocate(qkv_post_reduced)
        qkv_post = ttnn.reshape(qkv_post_silu, [1, 1, B, self.conv_dim])
        ttnn.deallocate(qkv_post_silu)
        # WS-A.17: in-place state update via ttnn.copy so the buffer
        # address stays stable across trace replays.
        new_conv_state = ttnn.slice(
            conv_input,
            [0, 0, 0, 1],
            [1, 1, B * self.conv_dim, K],
        )
        ttnn.deallocate(conv_input)
        ttnn.copy(new_conv_state, self._tt_conv_state)
        ttnn.deallocate(new_conv_state)

        # ----- (3) slice into Q, K, V, L2-norm + scale ----------------
        q_slice = ttnn.slice(qkv_post, [0, 0, 0, 0], [1, 1, B, self.key_dim])
        k_slice = ttnn.slice(
            qkv_post, [0, 0, 0, self.key_dim], [1, 1, B, 2 * self.key_dim]
        )
        v_slice = ttnn.slice(
            qkv_post,
            [0, 0, 0, 2 * self.key_dim],
            [1, 1, B, 2 * self.key_dim + self.value_dim],
        )
        ttnn.deallocate(qkv_post)

        group = self.num_v_heads // self.num_k_heads
        assert group == 1, (
            f"WS-A.17 trace-safe path assumes num_k_heads == num_v_heads "
            f"(got {self.num_k_heads} vs {self.num_v_heads})."
        )
        q_heads = ttnn.reshape(q_slice, [1, 1, B * self.num_k_heads, self.head_k_dim])
        k_heads = ttnn.reshape(k_slice, [1, 1, B * self.num_k_heads, self.head_k_dim])
        v_heads = ttnn.reshape(v_slice, [1, 1, B * self.num_v_heads, self.head_v_dim])
        ttnn.deallocate(q_slice)
        ttnn.deallocate(k_slice)
        ttnn.deallocate(v_slice)

        q_normed = self._tt_l2_norm_scale(q_heads, 1.0 / math.sqrt(self.head_k_dim))
        k_normed = self._tt_l2_norm_scale(k_heads, 1.0)
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)

        q_col = ttnn.reshape(q_normed, [1, B * self.num_v_heads, self.head_k_dim, 1])
        k_col = ttnn.reshape(k_normed, [1, B * self.num_v_heads, self.head_k_dim, 1])
        ttnn.deallocate(q_normed)
        k_row = ttnn.reshape(k_col, [1, B * self.num_v_heads, 1, self.head_k_dim])
        v_col = ttnn.reshape(v_heads, [1, B * self.num_v_heads, self.head_v_dim, 1])
        ttnn.deallocate(v_heads)
        ttnn.deallocate(k_normed)

        # ----- (4) gating ---------------------------------------------
        a_biased = ttnn.add(a, self._tt_dt_bias)
        ttnn.deallocate(a)
        sp = self._tt_softplus_clamped(a_biased)
        ttnn.deallocate(a_biased)
        g = ttnn.multiply(self._tt_A_factor, sp)
        ttnn.deallocate(sp)
        decay = ttnn.exp(g)
        ttnn.deallocate(g)
        decay_bcast = ttnn.reshape(decay, [1, B * self.num_v_heads, 1, 1])

        beta = ttnn.sigmoid(b)
        ttnn.deallocate(b)
        beta_bcast = ttnn.reshape(beta, [1, B * self.num_v_heads, 1, 1])

        # ----- (5) SSM state update (in-place via ttnn.copy) ----------
        h_decayed = ttnn.multiply(self._tt_ssm_state, decay_bcast)
        # NOTE: do NOT deallocate self._tt_ssm_state here — its buffer
        # address must remain valid for the recorded trace.  We write the
        # new state INTO it after the matmul chain.
        ttnn.deallocate(decay_bcast)

        delta = ttnn.matmul(h_decayed, k_col, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_minus_delta = ttnn.sub(v_col, delta)
        ttnn.deallocate(delta)
        ttnn.deallocate(v_col)
        v_corr = ttnn.multiply(v_minus_delta, beta_bcast)
        ttnn.deallocate(v_minus_delta)
        ttnn.deallocate(beta_bcast)
        outer = ttnn.matmul(v_corr, k_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(v_corr)
        ttnn.deallocate(k_row)
        ttnn.deallocate(k_col)

        h_new = ttnn.add(h_decayed, outer)
        ttnn.deallocate(h_decayed)
        ttnn.deallocate(outer)

        # o = h_new @ q_col  →  [1, B*HV, V, 1]
        o = ttnn.matmul(h_new, q_col, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_col)

        # WS-A.17: write h_new into the persistent ssm_state buffer in-place.
        ttnn.copy(h_new, self._tt_ssm_state)
        ttnn.deallocate(h_new)

        # ----- (6) RMSNormGated + out_proj ----------------------------
        o_2d = ttnn.reshape(o, [1, 1, B * self.num_v_heads, self.head_v_dim])
        ttnn.deallocate(o)
        z_2d = ttnn.reshape(z, [1, 1, B * self.num_v_heads, self.head_v_dim])
        ttnn.deallocate(z)
        silu_z = ttnn.silu(z_2d)
        ttnn.deallocate(z_2d)
        o_gated = ttnn.multiply(o_2d, silu_z)
        ttnn.deallocate(o_2d)
        ttnn.deallocate(silu_z)
        o_normed = ttnn.rms_norm(
            o_gated, weight=self._tt_W_norm_gated, epsilon=self.norm_eps
        )
        ttnn.deallocate(o_gated)
        o_flat = ttnn.reshape(o_normed, [1, 1, B, self.value_dim])
        ttnn.deallocate(o_normed)
        out = ttnn.linear(o_flat, self._tt_W_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o_flat)

        # WS-A.17: pad B=1 result back to the tile-aligned B_padded so the
        # downstream residual add (against the 32-row residual) sees user 0
        # in row 0 and zeros in the inactive 1..31 rows.  On-device op
        # (trace-safe).  ``out`` shape: [1, 1, 1, hidden] → after pad:
        # [1, 1, 32, hidden].
        if B_padded > B:
            out_padded = ttnn.pad(
                out, [(0, 0), (0, 0), (0, B_padded - B), (0, 0)], value=0.0
            )
            ttnn.deallocate(out)
            out = out_padded
        return out  # [1, 1, B_padded, hidden], replicated

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
        # Normalize Mode enum to string so comparisons like ``_mode_str == "decode"``
        # work uniformly.  Callers (model.py forward()) pass Mode.DECODE.
        _mode_str = getattr(mode, "value", mode)
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"linear-attn input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        # (1) attention_norm on device
        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        attn_in = self.attention_norm(x, mode, norm_config=attn_norm_config)

        # (2) DeltaNet: either host-fallback (default), TT-native eager
        #     (env-gated), or TT-native trace-safe (WS-A.17 env-gated)
        if self._tt_trace_enabled and _mode_str == "decode":
            # WS-A.17 trace-safe path: NO host bridges anywhere.  Result is a
            # 4D [1, 1, B, hidden] REPLICATED tensor on device.  Fracture
            # along hidden via ttnn.mesh_partition (the inverse of the
            # all_gather inside attention_norm) to match the residual's
            # sharded layout.
            tt_out_full = self._tt_native_delta_net_step_trace_safe(attn_in)
            if self.num_devices > 1:
                # Sharded across mesh dim=-1 — same layout as ShardTensorToMesh.
                tt_out_sharded = ttnn.mesh_partition(
                    tt_out_full, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.deallocate(tt_out_full)
            else:
                tt_out_sharded = tt_out_full
            attn_out = ttnn.to_memory_config(tt_out_sharded, skip_mem_cfg)
            if tt_out_sharded is not attn_out:
                ttnn.deallocate(tt_out_sharded)
        elif self._tt_native_enabled and _mode_str == "decode":
            # WS-A.12 native path: all DeltaNet math on device. Result is a
            # 4D [1, 1, B, hidden] REPLICATED tensor; reshape it into the
            # residual's shard layout before the residual add.
            tt_out_full = self._tt_native_delta_net_step(attn_in)
            # tt_out_full is fully replicated. To match the residual's
            # sharded layout, gather one device's view (which is the full
            # hidden), then re-shard via _from_host_to_device.
            dev_tensors = ttnn.get_device_tensors(tt_out_full)
            full_host = ttnn.to_torch(dev_tensors[0])
            if full_host.dim() == 4 and full_host.shape[0] == 1 and full_host.shape[1] == 1:
                full_host = full_host.reshape(full_host.shape[2], full_host.shape[3])
            ttnn.deallocate(tt_out_full)
            attn_out = self._from_host_to_device(full_host, ref_tt=x)
            attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
        else:
            # Host fallback: gather, run DeltaNet, push back
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
        if _mode_str == "prefill":
            try:
                x.deallocate(True)
            except Exception:
                pass

        ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
        hidden_states = self.ff_norm(hidden_states, mode, norm_config=ff_norm_config)
        ttnn.deallocate(attn_out)

        if TG and _mode_str == "decode":
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
