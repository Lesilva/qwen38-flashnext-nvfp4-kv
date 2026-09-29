"""GPU unit tests for QSA NVFP4 KV gathers (run inside the Pennyroyal 2.5.3 image
with the patched sglang tree mounted). Writes JSON results; exits non-zero on failure.

T1 layout: flashinfer quantize -> (a) flashinfer dequantize, (b) our independent
   PyTorch reference, (c) wrong layout hypotheses. (a)==(b) bit-exact; wrong ones far off.
T2 scale stats of the synthetic data (underflow/saturation of block scales).
T3 decode gather kernel vs torch reference: FP8 and BF16 scratch, -1 / out-of-range
   positions, rows beyond valid_count, page-stride packing, topk=2051 (not a
   multiple of 64), several batches.  Bit-exact on written rows.
T4 prefix gather kernel vs reference, bit-exact.
T5 timings: FP8 row copy vs NVFP4 dequant gather at decode (64 rows x 2051) and a
   512K-token prefix gather.
"""
import json, sys, time, pathlib
import torch

from sglang.srt.layers.attention.qsa.nvfp4_kv import (
    nvfp4_gather_dequant, qwen_sparse_kv_extraction_compact_nvfp4_triton, reference_dequant,
)
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton, qwen_sparse_valid_counts_triton,
)
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil
from flashinfer import nvfp4_kv_dequantize

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/nvfp4-unit.json")
dev = "cuda"; torch.manual_seed(0)
HEADS, DIM, PAGE, TOPK = 2, 256, 64, 2051
res = {}; ok = True


def check(name, cond, **info):
    global ok
    res[name] = {"pass": bool(cond), **info}; ok &= bool(cond)
    print(name, "PASS" if cond else "FAIL", json.dumps(info)[:300], flush=True)


def make_kv(m):
    # Per-16-group magnitudes spanning ~1e-2..64 so block scales vary widely.
    x = torch.randn(m, HEADS, DIM, device=dev)
    mag = torch.exp2(torch.empty(m, HEADS, DIM // 16, 1, device=dev).uniform_(-6, 6))
    return (x.view(m, HEADS, DIM // 16, 16) * mag).view(m, HEADS, DIM).to(torch.bfloat16)


def quantize(x, g):
    data, sf, _ = NVFP4KVQuantizeUtil.quantize(x.contiguous(), g)
    return data.view(torch.uint8), sf.view(torch.uint8)


g = torch.tensor([1.0], device=dev)
# ---- T1 layout
x = make_kv(512); data, sf = quantize(x, g)
# flashinfer's dequant only emits 16-bit outputs; compare in BF16 (fp32 math, one RNE cast on both sides).
fi = nvfp4_kv_dequantize(data.reshape(-1, DIM // 2), sf.reshape(-1, DIM // 16), g, output_dtype=torch.bfloat16).view(512, HEADS, DIM)
ref = reference_dequant(data, sf, g)
exact = torch.equal(fi.view(torch.int16), ref.to(torch.bfloat16).view(torch.int16))
rel = lambda a: ((a - x.float()).norm() / x.float().norm()).item()
swapped = data.clone(); swapped = ((swapped & 0x0F) << 4) | (swapped >> 4)
check("T1_layout", exact and rel(ref) < 0.15 and rel(reference_dequant(swapped, sf, g)) > 0.5,
      flashinfer_vs_ref_exact=exact, rel_err=rel(ref), rel_err_swapped_nibbles=rel(reference_dequant(swapped, sf, g)))
# ---- T2 scale stats
sfv = sf.view(torch.float8_e4m3fn).float()
check("T2_scale_stats", True, zero_scale_frac=(sfv == 0).float().mean().item(), max_scale=sfv.max().item(),
      note="synthetic data; real K/V stats come from the serving test")
# ---- pool for T3/T4
M = 200_000
xk, xv = make_kv(M), make_kv(M)
kd, ks = quantize(xk, g); vd, vs = quantize(xv, g)
gk = torch.tensor([1.0], device=dev); gv = torch.tensor([1.0], device=dev)
kref = reference_dequant(kd, ks, gk); vref = reference_dequant(vd, vs, gv)


def diff(got, exp):
    g, e = got.float(), exp.float()
    bad = (g != e)
    d = (g - e).abs()
    return {"mismatch_frac": bad.float().mean().item(), "max_abs": d.max().item(),
            "max_rel": (d / e.abs().clamp(min=1e-6)).max().item()}


def t3(batch, dtype):
    reqs = 32; ctx = 60_000
    req_to_token = torch.randint(1, M, (reqs, ctx), device=dev, dtype=torch.int32)
    req_idx = torch.randint(0, reqs, (batch,), device=dev, dtype=torch.int64)
    seq = torch.randint(TOPK // 2, ctx, (batch,), device=dev, dtype=torch.int32)
    idx = torch.randint(0, ctx, (batch, TOPK), device=dev, dtype=torch.int32)
    idx[:, -37:] = -1  # padded tail
    idx[0, 5] = ctx + 5  # beyond seq_len -> invalid
    stride = ((TOPK + PAGE - 1) // PAGE) * PAGE
    cu = torch.arange(batch + 1, device=dev, dtype=torch.int32) * stride
    ok_ = torch.empty(batch, device=dev, dtype=torch.int32)
    qwen_sparse_valid_counts_triton(seq, idx, ok_, batch, TOPK)
    ok_k = torch.randn((batch * stride, HEADS, DIM), device=dev).to(dtype)
    ok_v = torch.randn((batch * stride, HEADS, DIM), device=dev).to(dtype)
    before_k, before_v = ok_k.clone(), ok_v.clone()
    qwen_sparse_kv_extraction_compact_nvfp4_triton(kd, vd, ks, vs, gk, gv, req_to_token, req_idx, idx, seq, cu, ok_k, ok_v, batch, TOPK)
    torch.cuda.synchronize()
    # reference: same validity rule as _compact_kv
    cols = torch.arange(TOPK, device=dev)
    valid = (cols[None, :] < stride) & (idx >= 0) & (idx < seq[:, None])
    slots = req_to_token[req_idx[:, None], idx.clamp(0, ctx - 1).long()].long()  # invalid ones are masked below
    exp_k = kref[slots].to(dtype); exp_v = vref[slots].to(dtype)
    got_k = ok_k.view(batch, stride, HEADS, DIM)[:, :TOPK]; got_v = ok_v.view(batch, stride, HEADS, DIM)[:, :TOPK]
    b8 = lambda t: t.view(torch.uint8) if t.dtype != torch.bfloat16 else t.view(torch.int16)
    # Value equality (bit patterns may differ only in the sign of zero: -0.0 == 0.0).
    same = bool(torch.equal(got_k[valid].float(), exp_k[valid].float()) and torch.equal(got_v[valid].float(), exp_v[valid].float())
                and not torch.isnan(got_k[valid].float()).any())
    bk = before_k.view(batch, stride, HEADS, DIM)[:, :TOPK]
    untouched = torch.equal(b8(got_k)[~valid], b8(bk)[~valid])
    diag = diff(got_k[valid], exp_k[valid])
    return same, untouched, valid.sum().item(), diag


for dtype in (torch.float8_e4m3fn, torch.bfloat16):
    for batch in (1, 16, 64):
        same, untouched, nvalid, diag = t3(batch, dtype)
        check(f"T3_decode_gather_{str(dtype).split('.')[-1]}_b{batch}", same and untouched, valid_rows=nvalid, invalid_rows_untouched=untouched, **diag)

for dtype in (torch.float8_e4m3fn, torch.bfloat16):
    index = torch.randint(0, M, (70_001,), device=dev, dtype=torch.int64)
    got = nvfp4_gather_dequant(kd, ks, gk, index, dtype)
    exp = kref[index].to(dtype)
    b = lambda t: t.view(torch.uint8) if t.dtype != torch.bfloat16 else t.view(torch.int16)
    check(f"T4_prefix_gather_{str(dtype).split('.')[-1]}", torch.equal(got.float(), exp.float()) and not torch.isnan(got.float()).any(), rows=index.numel(), **diff(got, exp),
          rel_err_vs_original=((got.float() - xk[index].float()).norm() / xk[index].float().norm()).item())


def bench(fn, n=50):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3


# ---- T5 timings (FP8 baseline = current production row copy)
batch = 64; reqs = 16; ctx = 60_000
req_to_token = torch.randint(1, M, (reqs, ctx), device=dev, dtype=torch.int32)
req_idx = torch.randint(0, reqs, (batch,), device=dev, dtype=torch.int64)
seq = torch.full((batch,), ctx, device=dev, dtype=torch.int32)
idx = torch.randint(0, ctx, (batch, TOPK), device=dev, dtype=torch.int32)
stride = ((TOPK + PAGE - 1) // PAGE) * PAGE
cu = torch.arange(batch + 1, device=dev, dtype=torch.int32) * stride
fp8k = xk.to(torch.float8_e4m3fn); fp8v = xv.to(torch.float8_e4m3fn)
sk = torch.empty((batch * stride, HEADS, DIM), device=dev, dtype=torch.float8_e4m3fn); sv = torch.empty_like(sk)
t_fp8 = bench(lambda: qwen_sparse_kv_extraction_compact_triton(fp8k, fp8v, req_to_token, req_idx, idx, seq, cu, sk, sv, batch, TOPK))
t_nv = bench(lambda: qwen_sparse_kv_extraction_compact_nvfp4_triton(kd, vd, ks, vs, gk, gv, req_to_token, req_idx, idx, seq, cu, sk, sv, batch, TOPK))
index = torch.randint(0, M, (512 * 1024,), device=dev, dtype=torch.int64)
t_sel = bench(lambda: fp8k.index_select(0, index), 10)
t_gd = bench(lambda: nvfp4_gather_dequant(kd, ks, gk, index, torch.float8_e4m3fn), 10)
check("T5_timing", True, decode_gather_ms={"fp8_copy": t_fp8, "nvfp4_dequant": t_nv},
      prefix_gather_512k_ms_per_tensor={"fp8_index_select": t_sel, "nvfp4_dequant": t_gd},
      note="per layer; decode is 64 rows x 2051 (C16 x MTP4)")
res["all_pass"] = ok
out.write_text(json.dumps(res, indent=1)); print("ALL_PASS" if ok else "SOME_FAIL", flush=True)
sys.exit(0 if ok else 1)
