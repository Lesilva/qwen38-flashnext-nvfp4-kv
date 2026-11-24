"""Multi-CTA exact top-512 (drop-in for sglang fast_topk(score, lengths, 512, row_starts=None)).

Build: JIT through sglang.kernels.jit.utils.load_jit (absolute source path).
Workspace: one persistent, zero-initialised int32 [rows_cap, ROW_WORDS] buffer per device,
self-cleaning between calls. Call `mc_topk_prepare(max_rows, device)` before CUDA-graph
capture; growing it during capture raises. Calls sharing a workspace must be serialised on
one stream (they are in the decode path).

Barrier timeout: the kernel recovers by itself inside the same call (the last CTA of the row
resets the row's workspace and recomputes it exactly on one CTA); `mc_topk_timeout_count()`
reports how many such recoveries ever happened (sticky, host sync).
"""
from __future__ import annotations

import os

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mc_topk.cuh")
STATE = 8192 + 256  # must match mc_topk.cuh kState
ROW_WORDS = STATE + 16 + 2 * 8192  # kRowWords
ERR_WORD = STATE + 6  # kErrWord: sticky count of recovered barrier timeouts
DONE_WORD = STATE + 7  # kDone: per-call exit word (zero between calls)
GEN_WORDS = (STATE + 2, STATE + 3)
DEFAULT_BLOCK = int(os.environ.get("MC_TOPK_BLOCK", "512"))
DEFAULT_IPT = int(os.environ.get("MC_TOPK_IPT", "8"))


@cache_once
def _module(topk: int, block: int, ipt: int, src: str = _SRC, debug: bool = False):
    args = make_cpp_args(topk, block, ipt, debug)
    tag = os.path.splitext(os.path.basename(src))[0]
    return load_jit(
        tag,
        *args,
        cuda_files=[src],
        cuda_wrappers=[("mc_topk", f"McTopKKernel<{args}>::run")],
    )


_WS: dict = {}
_WS_RETIRED: list = []  # outgrown workspaces: kept alive, captured graphs may still reference them
_DUMMY: dict = {}


def _dev_key(device) -> tuple:
    device = torch.device(device if device is not None else "cuda")
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return device.type, idx


def mc_topk_prepare(max_rows: int, device=None, block: int = DEFAULT_BLOCK, ipt: int = DEFAULT_IPT) -> torch.Tensor:
    """Allocate/grow the zeroed workspace, the dummy trace tensor and JIT-load the module.
    Call before CUDA-graph capture with the largest row count that will be captured. Growing
    keeps the old buffer alive (graphs captured earlier keep pointing at it)."""
    key = _dev_key(device)
    dev = torch.device(key[0], key[1])
    ws = _WS.get(key)
    if ws is None or ws.shape[0] < max_rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mc_topk workspace must be prepared (mc_topk_prepare) before CUDA graph capture")
        n = 16
        while n < max_rows:
            n *= 2
        if ws is not None:
            _WS_RETIRED.append(ws)
        ws = torch.zeros((n, ROW_WORDS), dtype=torch.int32, device=dev)
        _WS[key] = ws
    if key not in _DUMMY:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mc_topk_prepare must run before CUDA graph capture")
        _DUMMY[key] = torch.zeros(1, dtype=torch.int64, device=dev)
        _module(512, block, ipt)
    return ws


def mc_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int = 512,
    *,
    chunk_target: int = 0,
    max_ctas_per_row: int = 0,
    cooperative: bool = True,
    block: int = DEFAULT_BLOCK,
    ipt: int = DEFAULT_IPT,
    workspace: torch.Tensor | None = None,
    trace: torch.Tensor | None = None,
    src: str | None = None,
    debug_ctl: torch.Tensor | None = None,
) -> torch.Tensor:
    """score fp32 [B, W] (unit stride in dim 1), lengths int32 [B] -> int32 [B, topk].

    debug_ctl (tests only): int64 [16] device tensor; selects the DBG build with fault
    injection (layout in mc_topk.cuh Params::dbg). Never pass it in production."""
    if topk != 512:
        raise RuntimeError("mc_topk only built for topk=512")
    B = score.shape[0]
    ws = workspace if workspace is not None else mc_topk_prepare(B, score.device)
    out = score.new_empty((B, topk), dtype=torch.int32)
    dummy = _DUMMY[_dev_key(score.device)]
    if trace is None:
        trace = dummy
    debug = debug_ctl is not None
    _module(topk, block, ipt, src or _SRC, debug).mc_topk(
        score, lengths, out, ws, trace, debug_ctl if debug else dummy,
        int(chunk_target), int(max_ctas_per_row), int(bool(cooperative)),
    )
    return out


def mc_topk_error_flag(device=None) -> int:
    """Number of workspace rows on which a barrier timeout ever happened (sticky; each was
    recovered in-kernel with an exact result). Host sync; diagnostics only."""
    ws = mc_topk_prepare(1, device)
    return int((ws[:, ERR_WORD] != 0).sum().item())


def mc_topk_timeout_count(device=None) -> int:
    """Total number of (row, call) barrier timeouts ever recovered (sticky). Host sync."""
    ws = mc_topk_prepare(1, device)
    return int(ws[:, ERR_WORD].long().sum().item())


def mc_topk_reset_workspace(device=None) -> None:
    """Re-zero the workspace in place (not needed after a barrier timeout any more; also
    clears the sticky timeout counters). Not during capture; the address stays the same so
    captured graphs remain valid."""
    mc_topk_prepare(1, device).zero_()
