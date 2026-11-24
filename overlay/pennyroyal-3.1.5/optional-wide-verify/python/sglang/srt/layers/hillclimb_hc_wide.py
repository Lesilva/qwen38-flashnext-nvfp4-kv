"""Local patch (2026-11-24, lever E', QWEN_EP_HC_WIDE=1): HC MIX2 with FP8 (E4M3, fp32 per-output-row scale) mix weights for 1..64 rows.

Same three Penny kernels (K0 `_hc_branch_stats_kernel`, K1 `_hc_down_kernel`,
K2 `_hc_up_kernel` from sglang.srt.layers.hc_mix2_triton), same math, same
weight/scale format as the resident online-FP8 Parameters; only the launcher
differs: ROWS is padded to 16/32/64 instead of a hard 16-row gate, and the
launch geometry is picked per ROWS bucket from GEOM (tuned by sweep.py).
"""
from __future__ import annotations

import triton
import torch

from sglang.srt.layers.hc_mix2_triton import (
    _hc_branch_stats_kernel,
    _hc_down_kernel,
    _hc_up_kernel,
    _STATS_BLOCK,
    _STATS_WARPS,
)

MAX_ROWS = 64
# Geometry per row bucket: key = bucket, ROWS (tl.dot M, >= 16) = max(16, bucket).
# (down: BLOCK_N, BLOCK_K, BLOCK_G, warps, stages), (up: BLOCK_J, BLOCK_R, warps, stages)
# Penny's pinned 16-row geometry is DOWN=(64,128,4,8,4) UP=(16,64,4,4).
# Values from sweep.py (CUDA-graph replay, 64 distinct FP8 weight copies), see
# sweep_*.json. rows<=16 keeps Penny's pinned geometry: the alternatives
# (a rows<=4 tile that won the isolated sweep, K2 at 8 warps) were within
# run-to-run noise or worse end-to-end (ab_small.json).
GEOM = {
    16: dict(down=(64, 128, 4, 8, 4), up=(16, 64, 4, 4)),
    32: dict(down=(64, 128, 4, 4, 4), up=(16, 64, 4, 4)),
    64: dict(down=(32, 128, 2, 8, 2), up=(16, 32, 8, 4)),
}
PENNY_GEOM = dict(down=(64, 128, 4, 8, 4), up=(16, 64, 4, 4))


def rows_bucket(rows: int) -> int:
    return 16 if rows <= 16 else (32 if rows <= 32 else 64)


_SEEN = set()


def hc_norm_mix2_wide(x, norm_w, eps, w_down, w_up, s_down, s_up, hc=4, hs=2560,
                      geom=None, use_pdl=False):
    rows, k = x.shape
    if rows not in _SEEN:
        _SEEN.add(rows)
        import logging

        logging.getLogger(__name__).info("hillclimb E' HC MIX2 wide: rows=%d bucket=%d", rows, rows_bucket(rows))
    assert 1 <= rows <= MAX_ROWS and k == hc * hs
    lowrank = w_down.shape[0]
    bucket = rows_bucket(rows)
    rp = max(16, bucket)
    g = geom or GEOM[bucket]
    bn, bk, bg, dw, ds = g["down"]
    bj, br, uw, us = g["up"]
    assert hs % (bk * bg) == 0 and lowrank % bn == 0
    mixed = torch.empty((rows, hs), dtype=x.dtype, device=x.device)
    normed = torch.empty_like(x)
    t_raw = torch.empty((rp, lowrank), dtype=torch.float32, device=x.device)
    num_tasks, zero_span = rows * hc, rows * lowrank
    _hc_branch_stats_kernel[(num_tasks,)](
        x, norm_w, normed, t_raw, num_tasks, zero_span, k, hs, eps,
        HC=hc, BLOCK_S=_STATS_BLOCK,
        ZERO_BLOCK=max(64, triton.next_power_of_2(zero_span // num_tasks)),
        USE_PDL=use_pdl, launch_pdl=use_pdl, num_warps=_STATS_WARPS)
    _hc_down_kernel[(k // (bk * bg), triton.cdiv(lowrank, bn))](
        w_down, normed, t_raw, s_down, rows, k, lowrank,
        ROWS=rp, BLOCK_N=bn, BLOCK_K=bk, BLOCK_G=bg, NUM_STAGES=ds,
        USE_PDL=use_pdl, launch_pdl=use_pdl, num_warps=dw)
    _hc_up_kernel[(triton.cdiv(hs, bj),)](
        normed, w_up, t_raw, mixed, s_up, rows, k, hs, lowrank, 1.0 / hc,
        ROWS=rp, HC=hc, BLOCK_J=bj, BLOCK_R=br, NUM_STAGES=us,
        USE_PDL=use_pdl, launch_pdl=use_pdl, num_warps=uw)
    return mixed, normed
