"""GPU tests for the QSA indexer prefill-MQA split-K patch (run inside the Pennyroyal image
with the patched mqa.py mounted). Writes JSON; exits non-zero on failure.

S1 bit-exactness: split-K logits == split=1 logits (torch.equal, incl. the -inf mask) on
   causal chunk shapes at 8K..480K depth and on a packed two-request batch.
S2 reference: small case against the torch reference implementation (allclose).
S3 timing: split=1 vs auto split per indexer row chunk at each depth, using the indexer's
   own row-chunk sizing (128 MiB logits budget), i.e. the shapes a 2048-token prefill
   chunk really launches.
"""
import json, os, sys, time, pathlib
import torch

import sglang.srt.layers.attention.qsa.mqa as mqa
from sglang.srt.layers.attention.qsa.qsa_indexer import _qsa_prefill_row_chunk_size

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/splitk.json")
dev = "cuda"; torch.manual_seed(0)
HEADS, DIM, RATIO, CHUNK = 4, 128, 4, 2048
res = {}; ok = True


def check(name, cond, **info):
    global ok
    res[name] = {"pass": bool(cond), **info}; ok &= bool(cond)
    print(name, "PASS" if cond else "FAIL", json.dumps(info)[:300], flush=True)


def causal_case(depth, rows):
    """`rows` query rows of one request whose first row sits at token `depth`."""
    keys = (depth + rows) // RATIO
    q = torch.randn(rows, HEADS, DIM, device=dev, dtype=torch.bfloat16)
    k = torch.randn(keys, 1, DIM, device=dev, dtype=torch.bfloat16)
    pos = torch.arange(depth, depth + rows, device=dev)
    ends = torch.clamp(pos // RATIO + 1, max=keys).to(torch.int32)
    starts = torch.zeros_like(ends)
    return q, k, starts, ends


def run(q, k, s, e, split):
    os.environ["SGLANG_QSA_MQA_SPLITK"] = str(split)
    return mqa.tilelang_qsa_mqa_prefill(q, k, s, e)


# S1: bit-exact on deep causal chunks (one indexer row-chunk each) and a packed batch.
for depth in [8192, 32768, 131072, 262144, 491520]:
    rows = _qsa_prefill_row_chunk_size(CHUNK, (depth + CHUNK) // RATIO, HEADS)
    q, k, s, e = causal_case(depth, rows)
    a = run(q, k, s, e, 1); b = run(q, k, s, e, "auto")
    split = mqa._qsa_mqa_prefill_split(rows + (-rows) % 32, k.shape[0], 32)
    check(f"S1_exact_depth{depth}", torch.equal(a, b), rows=rows, keys=k.shape[0], split=split,
          finite=int(torch.isfinite(a).sum()))
# packed: two requests with different windows (starts > 0 for the second), odd row count
q1, k1, s1, e1 = causal_case(40000, 777); q2, k2, s2, e2 = causal_case(9000, 555)
kk = torch.cat([k1, k2]); off = k1.shape[0]
q = torch.cat([q1, q2]); s = torch.cat([s1, s2 + off]); e = torch.cat([e1, e2 + off])
a = run(q, kk, s, e, 1)
for sp in ["auto", 2, 64]:
    b = run(q, kk, s, e, sp); check(f"S1_exact_packed_split{sp}", torch.equal(a, b), rows=q.shape[0], keys=kk.shape[0])

# S2: torch reference on a small case
q, k, s, e = causal_case(4096, 300)
ref = mqa.torch_qsa_mqa_prefill(q, k, s, e); got = run(q, k, s, e, "auto")
fin = torch.isfinite(ref)
check("S2_vs_torch", torch.equal(fin, torch.isfinite(got)) and torch.allclose(ref[fin], got[fin], rtol=2e-2, atol=2e-2),
      max_abs=float((ref[fin] - got[fin]).abs().max()))

# S3: timing per depth for the shapes one 2048-token prefill chunk launches
timing = {}
for depth in [8192, 32768, 131072, 262144, 491520]:
    keys = (depth + CHUNK) // RATIO
    rows = _qsa_prefill_row_chunk_size(CHUNK, keys, HEADS)
    q, k, s, e = causal_case(depth, rows)
    t = {}
    for sp in [1, "auto"]:
        for _ in range(2): run(q, k, s, e, sp)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5): run(q, k, s, e, sp)
        torch.cuda.synchronize(); t[str(sp)] = (time.perf_counter() - t0) / 5 * 1e3
    launches = -(-CHUNK // rows)
    timing[depth] = {"rows_per_launch": rows, "launches_per_chunk": launches, "ms_split1": t["1"], "ms_auto": t["auto"],
                     "speedup": t["1"] / t["auto"], "chunk_ms_split1": t["1"] * launches, "chunk_ms_auto": t["auto"] * launches}
    print("S3", depth, json.dumps(timing[depth]), flush=True)
# Gate: a real win at the deepest depth, and no depth more than 10% slower than split=1.
check("S3_timing", timing[491520]["speedup"] >= 1.5 and all(v["speedup"] >= 0.9 for v in timing.values()), timing=timing)

res["all_pass"] = ok
out.write_text(json.dumps(res, indent=1)); print("ALL_PASS" if ok else "SOME_FAIL", flush=True)
sys.exit(0 if ok else 1)
