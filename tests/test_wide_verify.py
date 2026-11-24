"""Module tests for the optional-wide-verify set on Pennyroyal 3.1.5 (levers C, E', I, J). Run them in a container
of the stock 3.1.5 image with the nvfp4-kv and optional-wide-verify overlays mounted (the -v lines printed by
`./examples/overlay-mounts.sh --with-wide-verify`), `-e SGLANG_SM120_ONLINE_MXFP8=true`, `--user 0:0`, this tests/
directory mounted at /tests, and `--entrypoint /opt/pennyroyal/.venv/bin/python ... /tests/test_wide_verify.py`.
No model is loaded; it needs about 1 GiB of VRAM and a minute. The test switches the four env flags itself.

usage: test_wide_verify.py [E I J C ...]   -> prints JSON lines, writes $OUT (default /tmp/test-wide-verify.json)
"""
import json, os, sys, types, time
import torch

dev = torch.device("cuda")
# hard in-process cap: PyTorch allocations above 700 MiB raise OOM here instead of eating production headroom
torch.cuda.set_per_process_memory_fraction(700 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
torch.manual_seed(0)
RES = []
FAIL = []


def rec(**kw):
    RES.append(kw)
    print(json.dumps(kw), flush=True)
    if not kw.get("ok", True):
        FAIL.append(kw)


def setenv(name, on):
    os.environ[name] = "1" if on else "0"


def graph_run(fn):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay(); g.replay(); torch.cuda.synchronize()
    return g, out


def maxabs(a, b):
    return float((a.float() - b.float()).abs().max())


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


import sglang.kernels.ops.gemm.sm120_online_fp8 as so

so._online_fp8_enabled = True
so._fast_paths_enabled = True
from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv, prealloc

prealloc(dev)
import sglang.kernels.ops.gemm.hillclimb_wide_gemm as hw

hw.prealloc(dev)


def mk_fp8(N, K, std=0.02):
    q = torch.empty(N, K, device=dev, dtype=torch.float8_e4m3fn)
    s = torch.empty(N, device=dev, dtype=torch.float32)
    for a in range(0, N, 8192):  # chunked: no full-size fp32 temporary
        w = torch.randn(min(8192, N - a), K, device=dev) * std
        sc = w.abs().amax(1).clamp(min=1e-8) / 448.0
        q[a:a + w.shape[0]] = (w / sc[:, None]).to(torch.float8_e4m3fn)
        s[a:a + w.shape[0]] = sc
        del w
    return q, s


def chunked(fn, x):
    return torch.cat([fn(x[i:i + 16].contiguous()) for i in range(0, x.shape[0], 16)], 0)


# ------------------------------------------------------------------------------------------------ J
def test_J():
    from sglang.srt.models import qwen4_exp as q4

    captured = {}

    def stub(model, *, enabled, method_factory, **kw):
        captured["f"] = method_factory
        return []

    orig = so.convert_eligible_linears_to_mxfp8
    so.convert_eligible_linears_to_mxfp8 = stub
    try:
        q4.Qwen4ExpLayerExtensionMixin._maybe_convert_linears_to_mxfp8(types.SimpleNamespace())
    finally:
        so.convert_eligible_linears_to_mxfp8 = orig
    method = captured["f"]()
    rec(test="J.method", cls=type(method).__name__, use_marlin=method.use_marlin, block_quant=method.block_quant,
        use_mxfp8=method.use_mxfp8, ok=not method.use_marlin)
    shapes = {"gdn_qkvz": (16384, 2560), "gdn_out": (2560, 6144), "attn_qkv": (13312, 2560),
              "attn_o": (2560, 6144), "shared_gate_up": (1280, 2560), "shared_down": (2560, 640)}
    for name, (N, K) in shapes.items():
        qw, s = mk_fp8(N, K)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(qw.t(), requires_grad=False)  # [K, N] view, as Penny stores it
        layer.weight_scale = torch.nn.Parameter(s.clone(), requires_grad=False)
        layer.input_scale = None
        for M in (4, 16, 20, 24, 32, 48, 64):
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            ref = chunked(lambda c: w8a16_gemv(c, qw, s), x)
            exact = (x.float() @ qw.float().t()) * s[None, :]
            setenv("QWEN_J_WIDE_FP8LIN", True)
            hw._SEEN.clear()
            y_on = method.apply(layer, x)
            used = any(k[0] == "J" and k[3] == M for k in hw._SEEN)
            g, y_g = graph_run(lambda: method.apply(layer, x))
            g_same = torch.equal(y_g, y_on)
            del g
            setenv("QWEN_J_WIDE_FP8LIN", False)
            hw._SEEN.clear()
            try:
                y_off = method.apply(layer, x)
                off_err = None
            except Exception as e:
                y_off, off_err = None, repr(e)[:120]
            used_off = bool(hw._SEEN)
            row = dict(test="J", shape=name, M=M, wide_used=used, wide_used_when_off=used_off,
                       maxabs_vs_chunked16=maxabs(y_on, ref), rel_vs_exact=rel(y_on, exact),
                       graph_equals_eager=g_same)
            if y_off is not None:
                row["off_rel_vs_exact"] = rel(y_off, exact)
            if off_err:
                row["off_err"] = off_err
            want_used = M > 16
            row["ok"] = (used == want_used and not used_off and g_same and row["rel_vs_exact"] < 5e-3
                         and (M <= 16 or row["maxabs_vs_chunked16"] <= 0.0625 * float(ref.abs().max())))
            rec(**row)
        del qw, s, layer
        torch.cuda.empty_cache()
    setenv("QWEN_J_WIDE_FP8LIN", False)


# ------------------------------------------------------------------------------------------------ I
def test_I():
    for N in (98304, 98304 - 5):
        qw, s = mk_fp8(N, 2560)
        w = torch.nn.Parameter(qw, requires_grad=False)
        setattr(w, so._SCALE_ATTR, s)
        for M in (4, 16, 20, 24, 32, 48, 64):
            h = torch.randn(M, 2560, device=dev, dtype=torch.bfloat16)
            ref = chunked(lambda c: w8a16_gemv(c, qw, s), h)
            setenv("QWEN_I_WIDE_LMHEAD", True)
            hw._SEEN.clear()
            y_on = so.rowwise_fp8_lm_head_logits(h, w)
            used = any(k[0] == "I" for k in hw._SEEN)
            g, y_g = graph_run(lambda: so.rowwise_fp8_lm_head_logits(h, w))
            g_same = torch.equal(y_g, y_on)
            del g
            setenv("QWEN_I_WIDE_LMHEAD", False)
            hw._SEEN.clear()
            y_off = so.rowwise_fp8_lm_head_logits(h, w)
            used_off = bool(hw._SEEN)
            row = dict(test="I", N=N, M=M, wide_used=used, wide_used_when_off=used_off,
                       maxabs_vs_chunked16=maxabs(y_on, ref), bit_equal_chunked16=torch.equal(y_on, ref),
                       maxabs_off_vs_chunked16=maxabs(y_off, ref), graph_equals_eager=g_same)
            row["ok"] = used == (M > 16) and not used_off and g_same and row["maxabs_vs_chunked16"] <= 0.02 * float(ref.abs().max())
            rec(**row)
        del qw, s, w
        torch.cuda.empty_cache()
    setenv("QWEN_I_WIDE_LMHEAD", False)


# ------------------------------------------------------------------------------------------------ E'
def test_E():
    import sglang.srt.layers.hc_mix2_triton as hm
    import sglang.srt.layers.hillclimb_hc_wide as hcw

    hc, hs, lr = 4, 2560, 320
    k = hc * hs
    wd, sd = mk_fp8(lr, k, 0.01)
    wu, su = mk_fp8(k, lr, 0.05)
    pd = torch.nn.Parameter(wd, requires_grad=False); setattr(pd, so._SCALE_ATTR, sd)
    pu = torch.nn.Parameter(wu, requires_grad=False); setattr(pu, so._SCALE_ATTR, su)
    norm_w = (torch.randn(k, device=dev) * 0.1).to(torch.bfloat16)
    eps = 1e-6
    for M in (4, 16, 17, 20, 24, 32, 33, 48, 64):
        x = (torch.randn(M, k, device=dev) * torch.linspace(0.5, 4, M, device=dev)[:, None]).to(torch.bfloat16)

        def call(xx):
            return hm.hc_norm_mix2(xx, norm_w, eps, pd, pu, hc, hs)

        setenv("QWEN_EP_HC_WIDE", True)
        sup_on = hm.hc_norm_mix2_supported(x, norm_w, pd, pu, hc, hs)
        hcw._SEEN.clear()
        mixed, normed = call(x)
        used = M in hcw._SEEN
        rm = chunked(lambda c: call(c)[0], x)
        rn = chunked(lambda c: call(c)[1], x)
        g, (gm, gn) = graph_run(lambda: call(x))
        g_ok = maxabs(gm, mixed) <= 0.0625 and torch.equal(gn, normed)
        del g
        setenv("QWEN_EP_HC_WIDE", False)
        sup_off = hm.hc_norm_mix2_supported(x, norm_w, pd, pu, hc, hs)
        row = dict(test="E", M=M, supported_on=sup_on, supported_off=sup_off, wide_used=used,
                   normed_bit_equal=torch.equal(normed, rn), mixed_maxabs_vs_chunked16=maxabs(mixed, rm),
                   mixed_rel_vs_chunked16=rel(mixed, rm), graph_ok=g_ok)
        row["ok"] = (sup_on and sup_off == (M <= 16) and used == (M > 16) and row["normed_bit_equal"]
                     and row["mixed_rel_vs_chunked16"] < 5e-3 and g_ok)
        rec(**row)
    setenv("QWEN_EP_HC_WIDE", False)


# ------------------------------------------------------------------------------------------------ C
def test_C():
    import sglang.srt.layers.attention.qsa.qsa_indexer as qi
    from sglang.kernels.ops.elementwise.fast_topk import fast_topk
    from sglang.kernels.ops.elementwise import mc_topk as mc

    W = 131072
    box = {}
    orig_dec, orig_exp = qi.qsa_mqa_decode, qi.expand_qsa_block_indices
    qi.qsa_mqa_decode = lambda *a, **k: box["logits"]
    qi.expand_qsa_block_indices = lambda bi, *a, **k: bi
    fake = types.SimpleNamespace(block_topk=512, compress_ratio=4, token_topk=2048)
    try:
        for rows, L in ((1, 58500), (4, 1024), (4, 125000), (16, 58500), (64, 125000), (68, 30000)):
            lens = torch.tensor([max(1, L - (r % 4)) for r in range(rows)], device=dev, dtype=torch.int32)
            logits = torch.full((rows, W), float("-inf"), device=dev)
            for r in range(rows):
                logits[r, : int(lens[r])] = torch.randn(int(lens[r]), device=dev)
            box["logits"] = logits

            def call():
                return qi.QSAIndexer.select_decode_tokens(fake, None, None, None, lens, 524288, None, None)

            setenv("QWEN_C_MC_TOPK", True)
            qi._MC_TOPK_LOGGED.clear()
            y_on = call()
            used = rows in qi._MC_TOPK_LOGGED
            g, y_g = graph_run(call)
            # mutate inputs in place, replay: the graph must follow the new values/lengths
            for r in range(rows):
                logits[r, : int(lens[r])] = torch.randn(int(lens[r]), device=dev)
            g.replay(); torch.cuda.synchronize()
            ref2 = fast_topk(logits, lens, 512, row_starts=None)
            del g
            setenv("QWEN_C_MC_TOPK", False)
            qi._MC_TOPK_LOGGED.clear()

            def same_sets(a, b):
                bad = 0
                for r in range(rows):
                    sa = set(a[r][a[r] >= 0].tolist()); sb = set(b[r][b[r] >= 0].tolist())
                    if sa != sb:
                        # ties at the threshold may legitimately differ: compare value multisets
                        va = torch.sort(logits[r, list(sa)], descending=True).values if sa else None
                        vb = torch.sort(logits[r, list(sb)], descending=True).values if sb else None
                        if va is None or vb is None or not torch.equal(va, vb):
                            bad += 1
                return bad

            row = dict(test="C", rows=rows, L=L, mc_used=used, bad_rows_graph_replay=same_sets(y_g, ref2))
            box["logits"] = logits
            y_off = call()
            row["off_matches_fast_topk"] = same_sets(y_off, ref2) == 0
            row["err_flag"] = mc.mc_topk_error_flag(dev)
            row["ok"] = (used == (rows <= 64) and row["bad_rows_graph_replay"] == 0 and row["off_matches_fast_topk"]
                         and row["err_flag"] == 0)
            rec(**row)
    finally:
        qi.qsa_mqa_decode, qi.expand_qsa_block_indices = orig_dec, orig_exp
        setenv("QWEN_C_MC_TOPK", False)


if __name__ == "__main__":
    which = sys.argv[1:] or ["E", "I", "J", "C"]
    t0 = time.time()
    for t in which:
        try:
            globals()["test_" + t]()
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec(test=t, ok=False, error=repr(e)[:300])
        torch.cuda.empty_cache()
    summary = dict(tests=which, cases=len(RES), failed=len(FAIL), seconds=round(time.time() - t0, 1),
                   peak_alloc_MiB=round(torch.cuda.max_memory_allocated() / 2**20))
    print("SUMMARY", json.dumps(summary), flush=True)
    json.dump({"summary": summary, "results": RES}, open(os.environ.get("OUT", "/tmp/test-wide-verify.json"), "w"), indent=1)
    sys.exit(1 if FAIL else 0)
