"""GPU unit test for qfuse_ext.gemv_bf16 (needs a GPU; a few seconds). Prints one JSON line; exit 0 on pass."""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["QFUSE"] = "off"
import qfuse  # noqa: E402

e = qfuse.ext()
torch.manual_seed(0)
out = {"cases": [], "pass": True}


def graph_time(fn, reps=50, iters=7):
    for _ in range(3):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / reps)
    return sorted(ts)[len(ts) // 2]


for N in (96, 512):
    w = (torch.randn(N, 2560, device="cuda") * 0.05).to(torch.bfloat16)
    for T in (1, 2, 4, 8):
        x = torch.randn(T, 2560, device="cuda").to(torch.bfloat16)
        y1 = e.gemv_bf16(x, w)
        y2 = e.gemv_bf16(x, w)
        exact = (x.double() @ w.double().t())
        ref_bf = exact.to(torch.bfloat16)
        stock = torch.nn.functional.linear(x, w)
        rel = float((y1.double() - exact).norm() / exact.norm())
        rel_stock = float((stock.double() - exact).norm() / exact.norm())
        c = {
            "N": N, "T": T, "rel_vs_exact": rel, "stock_rel_vs_exact": rel_stock,
            "bit_equal_to_rounded_exact": float((y1 == ref_bf).float().mean()),
            "deterministic": bool(torch.equal(y1, y2)),
            "us_qfuse": graph_time(lambda: e.gemv_bf16(x, w)),
            "us_stock": graph_time(lambda: torch.nn.functional.linear(x, w)),
        }
        c["pass"] = c["deterministic"] and rel < 5e-3 and tuple(y1.shape) == (T, N)
        out["pass"] = out["pass"] and c["pass"]
        out["cases"].append(c)
print(json.dumps(out))
sys.exit(0 if out["pass"] else 1)
