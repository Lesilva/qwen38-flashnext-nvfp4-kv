"""Local patch (2026-11-24, levers I and J): rowwise-FP8 W8A16 skinny GEMM for 16 < M <= 64 on SM120.

Used by the dense online-FP8 linears (lever J, QWEN_J_WIDE_FP8LIN=1, qwen4_exp.py) and the rowwise-FP8
lm_head / FR-Spec hot head (lever I, QWEN_I_WIDE_LMHEAD=1, sm120_online_fp8.py).

    y[M, N] = bf16( (x[M, K] @ W[N, K]^T) * scale[N] ),  W fp8 e4m3 [N, K] K-contiguous (the resident
    online-FP8 weight exactly as Penny stores it: layer.weight is the [K, N] view, .t() gives [N, K]).

Penny's ``w8a16_gemv`` stops at M = 16. Above that, ``Fp8LinearMethod.apply`` falls back to the CUTLASS
W8A8 path with per-token activation quantization, which costs ~40 us per call on SM120 whatever the
shape (2560x6144 o_proj: 7.7 us at M=16 -> 39.8 us at M=20). NEXTN verify runs at M = 4*bs, so every
batch of 5..16 requests pays that on every dense linear (~3 ms per step).

Two kernel families, chosen per (N, K, M bucket) by a measured table:
  * "tile": an M-tiled kernel (one CTA per (M tile, N block), M tile fastest so CTAs sharing a weight
    tile run back to back and hit L2). No split-K, no scratch, nothing to prealloc, no stream hazard.
  * "splitk": Penny's own ``_w8a16_gemv_kernel`` (imported, unmodified) launched with M_PAD 32/64 and a
    private split-K scratch sized for M_PAD 64 (two slots, honouring Penny's ``scratch_slot`` choice,
    because the shared expert may run on an alternate stream). Must be prealloc'd before graph capture.
Numerics in both: bf16 operands, fp32 accumulation, acc * scale[n] rounded to bf16 -- Penny's epilogue.
"""

import functools
import os

import torch
import triton
import triton.language as tl

MIN_M = 17
MAX_M = 64


def _flag(name):
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def j_enabled() -> bool:
    return _flag("QWEN_J_WIDE_FP8LIN")


def i_enabled() -> bool:
    return _flag("QWEN_I_WIDE_LMHEAD")


# ----------------------------------------------------------------------------------------------
# family "tile"
# ----------------------------------------------------------------------------------------------
@triton.jit
def _w8a16_mtile_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_wn, stride_ym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    rows = tl.minimum(offs_n, N - 1)
    w_rows = w_ptr + rows.to(tl.int64)[:, None] * stride_wn
    x_rows = x_ptr + offs_m[:, None] * stride_xm
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        if EVEN_K:
            w = tl.load(w_rows + kk[None, :])
            x = tl.load(x_rows + kk[None, :], mask=m_mask[:, None], other=0.0)
        else:
            k_mask = kk < K
            w = tl.load(w_rows + kk[None, :], mask=k_mask[None, :], other=0.0)
            x = tl.load(x_rows + kk[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    scale = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
    out = (acc * scale[None, :]).to(tl.bfloat16)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], out,
             mask=m_mask[:, None] & n_mask[None, :])


def _launch_tile(x, w, s, y, cfg):
    _, bm, bn, bk, warps, stages = cfg
    M, K = x.shape
    N = w.shape[0]
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    _w8a16_mtile_kernel[grid](
        x, w, s, y, M, N, K, x.stride(0), w.stride(0), y.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, EVEN_K=K % bk == 0, num_warps=warps, num_stages=stages,
    )


# ----------------------------------------------------------------------------------------------
# family "splitk": Penny's kernel, private scratch
# ----------------------------------------------------------------------------------------------
_WS_FLOATS = 1 << 22  # 16 MiB per slot
_WS_COUNTERS = 4096
_WS = {}


def _scratch(device, slot):
    key = (device, slot)
    ws = _WS.get(key)
    if ws is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("w8a16_wide split-K scratch must be prealloc'd before CUDA graph capture")
        ws = (torch.empty(_WS_FLOATS, dtype=torch.float32, device=device),
              torch.zeros(_WS_COUNTERS, dtype=torch.int32, device=device))
        _WS[key] = ws
    return ws


def prealloc(device) -> None:
    from sglang.srt.layers.quantization import w8a16_gemv as g

    for slot in range(g._N_SLOTS):
        _scratch(device, slot)


def _splitk_fits(N, cfg):
    _, bn, bk, splits, warps, stages, m_pad = cfg
    nb = triton.cdiv(N, bn)
    return splits == 1 or (nb <= _WS_COUNTERS and nb * splits * m_pad * bn <= _WS_FLOATS)


def _launch_splitk(x, w, s, y, cfg):
    from sglang.kernels.triton_pdl import PDL
    from sglang.srt.layers.quantization import w8a16_gemv as g

    _, bn, bk, splits, warps, stages, m_pad = cfg
    M, K = x.shape
    N = w.shape[0]
    assert M <= m_pad and _splitk_fits(N, cfg)
    ws, cnt = _scratch(x.device, g._SLOT)
    g._w8a16_gemv_kernel[(triton.cdiv(N, bn), splits)](
        x, w, s, y, y, ws, cnt, x, x, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1), y.stride(0),
        x.stride(0), 0.0,
        SPLIT_N=0, PER_CHANNEL=True, BLOCK_N=bn, BLOCK_K=bk, M_PAD=m_pad, SPLITS=splits,
        EVEN_K=K % bk == 0, W_KN=False, USE_DOT=True, USE_PDL=PDL, SEG_NB0=0, NORM_G=0,
        NORM_SIGMOID=False, launch_pdl=PDL, num_warps=warps, num_stages=stages,
    )


# ----------------------------------------------------------------------------------------------
# plans
# ----------------------------------------------------------------------------------------------
def m_bucket(M: int) -> int:
    return 32 if M <= 32 else (48 if M <= 48 else 64)


# (N, K, M bucket) -> cfg; filled from j_sweep.py results (RTX PRO 6000, SM120, 188 SMs, cold L2).
# tile cfg:   ("tile", BLOCK_M, BLOCK_N, BLOCK_K, warps, stages)
# splitk cfg: ("splitk", BLOCK_N, BLOCK_K, SPLITS, warps, stages, M_PAD)
TUNED = {
    # lever I (d_head/lever_i sweep, 98304-row slice, 144 MiB L2 flush): target head and FR-Spec hot head
    (248320, 2560, 32): ("tile", 16, 64, 256, 4, 3),
    (248320, 2560, 48): ("tile", 16, 64, 256, 4, 3),
    (248320, 2560, 64): ("tile", 64, 128, 128, 8, 3),
    (98304, 2560, 32): ("tile", 16, 64, 256, 4, 3),
    (98304, 2560, 48): ("tile", 16, 64, 256, 4, 3),
    (98304, 2560, 64): ("tile", 64, 128, 128, 8, 3),
    # lever J (jlin/sweep-<shape>-<32|64>.json, 2026-11-24; 48-row calls use the 64 entry). Same-config
    # run-to-run spread on the 2560x6144 shape was up to ~25%, so these are "clearly better than ~40 us", not
    # fine-tuned optima. A ("off",) entry keeps the current CUTLASS W8A8 path for that shape/bucket.
    (2560, 6144, 32): ("tile", 16, 32, 256, 4, 3),            # GDN out_proj / attn o_proj   14.3 us
    (2560, 6144, 64): ("splitk", 32, 256, 2, 4, 3, 64),       # 14.7-20.1 us (tile 19.5-19.9)
    (16384, 2560, 32): ("tile", 16, 64, 128, 4, 3),           # GDN in_proj_qkvz             29.4 us
    (16384, 2560, 64): ("splitk", 128, 128, 1, 8, 3, 64),     # 31.9 us (tile 38.8)
    (13312, 2560, 32): ("tile", 16, 64, 128, 4, 3),           # attention qkv                24.6 us
    (13312, 2560, 64): ("tile", 64, 128, 128, 8, 3),          # 33.9 us (splitk 33.3)
    (1280, 2560, 32): ("tile", 16, 32, 256, 4, 3),            # shared-expert gate_up         6.1 us
    (1280, 2560, 64): ("tile", 16, 32, 256, 4, 3),            #                               6.3 us
    (2560, 640, 32): ("tile", 16, 32, 256, 4, 3),             # shared-expert down            3.4 us
    (2560, 640, 64): ("tile", 16, 32, 128, 4, 3),             #                               3.8 us
}


@functools.lru_cache(maxsize=256)
def plan(M: int, N: int, K: int):
    mb = m_bucket(M)
    cfg = TUNED.get((N, K, mb)) or (TUNED.get((N, K, 64)) if mb == 48 else None)
    if cfg is not None:
        return cfg
    # untuned shape: the scratch-free tile family with a generic tile
    return ("tile", 16, 64, 128, 4, 3)


def supported(x, w, scale) -> bool:
    return (
        x.dim() == 2 and MIN_M <= x.shape[0] <= MAX_M and x.dtype is torch.bfloat16 and x.stride(1) == 1
        and w.dim() == 2 and w.dtype is torch.float8_e4m3fn and w.stride(1) == 1 and w.shape[1] == x.shape[1]
        and scale.dtype is torch.float32 and scale.numel() == w.shape[0] and scale.is_contiguous()
        and x.device == w.device
        and plan(x.shape[0], w.shape[0], w.shape[1])[0] != "off"
    )


_SEEN = set()


def w8a16_gemv_wide(x, w, scale, cfg=None, out=None, tag="?"):
    """x [M, K] bf16 (17 <= M <= 64); w fp8 [N, K] K-contiguous; scale fp32 [N] -> bf16 [M, N]."""
    M, K = x.shape
    N = w.shape[0]
    s = scale.reshape(-1)
    y = out if out is not None else torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if cfg is None:
        cfg = plan(M, N, K)
        key = (tag, N, K, M)
        if key not in _SEEN:
            _SEEN.add(key)
            import logging

            logging.getLogger(__name__).info("hillclimb %s wide W8A16: N=%d K=%d M=%d cfg=%s", tag, N, K, M, cfg)
    if cfg[0] == "tile":
        _launch_tile(x, w, s, y, cfg)
    else:
        _launch_splitk(x, w, s, y, cfg)
    return y
