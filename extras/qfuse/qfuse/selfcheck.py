"""Shared error bookkeeping for selftest mode and the one-time per-instance check."""
from . import STATE, dump


def rel_err(fast, ref):
    d = (fast.float() - ref.float()).norm()
    return float(d / ref.float().norm().clamp_min(1e-12)), float((fast.float() - ref.float()).abs().max())


def record(kernel, rows, fast, fast2, ref, limit=256):
    st = STATE["selftest"].setdefault(
        "%s/rows=%d" % (kernel, rows),
        {"n": 0, "rel_l2": [], "max_abs": 0.0, "deterministic": True, "bit_equal": 0},
    )
    if st["n"] >= limit:
        return
    import torch

    r, m = rel_err(fast, ref)
    st["n"] += 1
    st["rel_l2"].append(r)
    st["max_abs"] = max(st["max_abs"], m)
    st["deterministic"] = bool(st["deterministic"] and torch.equal(fast, fast2))
    st["bit_equal"] += int(torch.equal(fast, ref))
    dump()
