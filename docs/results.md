# Results

Setup:
- Hardware: 1× RTX PRO 6000 Blackwell 96 GB (SM120), power cap 350 W, 89 GiB host RAM.
- Checkpoint: `jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4` (71.1 GiB weights, PLE tables offloaded to host RAM).
- Serving: 524,288 context (YaRN ×2), 16 running requests, MTP (NEXTN) 3 steps / 4 draft tokens, page 64, chunked prefill 2048.

Each A/B was run back to back on the same image, with production stopped.

## 1. Kernels (per layer, T5 in `tests/test_nvfp4_kv.py`)

| Gather | FP8 row copy (stock) | NVFP4 dequant gather |
|---|---:|---:|
| Decode: 64 rows × top-k 2051 | 0.166 ms | **0.145 ms** |
| Prefix: 512K tokens | 0.312 ms | **0.266 ms** |

NVFP4 reads fewer bytes, so the dequantizing gather is faster than the plain FP8 copy.

## 2. FP8 vs NVFP4, same 1M-token pool (Pennyroyal 2.5.2)

| Metric | FP8 | NVFP4 |
|---|---:|---:|
| Functional suite (22 chat tests: tools, JSON, code, images, 250K retrieval) | pass | pass |
| Target KV pool size | 12.16 GiB | **7.15 GiB** |
| Free VRAM after CUDA graphs | ~2.3 GiB | **~7.35 GiB** |
| Decode throughput, geomean of 24 cases (4 workloads × C1/2/3/4/8/16) | 1.000 | 0.989 (min 0.957) |
| MTP acceptance length | 2.297 | 2.315 |
| Cold prefill 8K / 32K / 128K / 240K | 11.1 / 11.1 / 10.5 / 9.7 K tok/s | −0.6 / −0.4 / −0.2 / −0.2 % |
| 490K three-needle retrieval | pass | pass |
| Stress (768 agent-style requests, cold injections) | 0 errors | 0 errors |

## 3. Spending the freed memory

We ran NVFP4 with `--prefill-decode-interval 1` and a 1.5M-token pool (1,499,968 after 64-page alignment), and compared it with the previous FP8 production config (1M pool, no interleave).

| Metric | FP8, 1M, interval 0 | NVFP4, 1.5M, interval 1 |
|---|---:|---:|
| KV pool | 1,000,000 tokens | **1,499,968 tokens** (10.73 GiB) |
| Decode throughput, 24-case geomean | 1.000 | 0.983 (min 0.954) |
| Longest stall of other streams during a 128K prefill | 12.2 s | **0.24 s** |
| 480K main-agent cold prefill: longest subagent stall | 67.5 s | **0.51 s** |
| 480K main-agent TTFT | 69.0 s | 79.3 s |
| Warm follow-up TTFT (480K cached) | 1.48 s | 1.48 s |
| Minimum free VRAM during the full suite | 1193 MiB | **2045 MiB** |

The +10 s main-agent TTFT is the cost of interleaving: other streams keep decoding while the long prefill runs.

Aggregate decode throughput of the final config (SparkDash harness: prose, 400 tokens, thinking off), in tok/s:

| Streams | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tok/s | 258 | 365 | 608 | 655 | 818 | 883 | 1092 |

## 4. Quality

We measured tail NLL of the last 128 prompt tokens on 24 distinct real zh/en text windows per length, with the same windows in both arms (paired). Values are in nats with 95% CIs.

| Context | FP8 → NVFP4 Δ NLL | Relative |
|---|---:|---:|
| 8K | +0.004 ± 0.011 | +0.3 % |
| 32K | +0.013 ± 0.047 | +1.0 % |
| 128K | +0.008 ± 0.023 | +0.6 % |

All intervals include 0. An FP8 → BF16 scratch buffer did not change the result. For reference, the vLLM community reported +0.37 … +0.59 % PPL for NVFP4 KV on the same model.

An earlier single-window measurement had shown 32K at +2–4 %. The paired 24-window run shows that was noise.

## 5. Pennyroyal 2.5.3 rebase

We compared 2.5.3 with the patches against 2.5.2 with the patches, both NVFP4 with the 1.5M pool.

| Metric | Result |
|---|---|
| Unit tests T1–T5; upstream `test_ngram_mamba_verify_update` | all pass |
| Functional (22) / Responses API | pass |
| Decode throughput vs 2.5.2, 24-case geomean | 1.010 (min 0.969) |
| Paired NLL vs 2.5.2 (8K / 32K / 128K) | −0.000 ± 0.013 / +0.016 ± 0.039 / +0.003 ± 0.046 |
| 490K retrieval / stress 768 requests | pass / 0 errors |
| Minimum free VRAM during the suite | 1987 MiB |

## 6. Pennyroyal 3.0.1 rebase (production since 2026-10-12)

The `qwen_sparse_attn_backend.py` merge had one conflict, a trivial init block. The RecoverSSM budget set was retired, since 3.0 ships the upstream fix. The configuration matches the production setup above, plus `FLASHINFER_GDN_FP16_ACCUM_MMA=1`, which 3.0's own launcher sets by default.

| Metric | 2.5.3 (prev. prod) | **3.0.1, fixed MTP (A, now prod)** | 3.0.1, adaptive MTP (B) |
|---|---:|---:|---:|
| Unit tests T1–T5 (FlashInfer 0.7.0.post1) | pass | pass | — |
| Functional / Responses / stress / concurrency | pass | pass | pass |
| Decode throughput vs 2.5.3, 24-case geomean | 1.000 | **1.056** (C1 +13–16%, C2 +13–18%, C4 +8%, C8+ ±3%) | — |
| C1 vs A, geomean | — | 1.000 | 0.970 |
| C1 decode, tok/s (zh / en / code / tool) | — | 227 / 222 / 248 / 236 | 221 / 207 / 254 / 226 |
| Cold prefill 8K / 32K / 128K / 240K | 11.1 / 11.1 / 10.6 / 9.7 K tok/s | 12.5 / 12.6 / 11.9 / 10.9 K | 11.6 / 12.6 / 11.9 / 10.8 K |
| 490K three-needle TTFT | 71.1 s | 66.5 s | 66.6 s |
| 480K main-agent TTFT / longest subagent stall | 79.0 s / 0.51 s | 74.8 s / 0.50 s | 75.0 s / 1.33 s |
| Paired NLL vs 2.5.3 (8K / 32K / 128K) | — | −0.008±0.024 / −0.041±0.054 / −0.014±0.030 | (vs A) −0.006±0.009 / −0.020±0.024 / −0.007±0.045 |
| Minimum free VRAM during the suite | 1987 MiB | **2749 MiB** | 1885 MiB |

All NLL intervals include 0. Adaptive MTP raised C1 acceptance from 2.62 to 2.72, but verifying 8 tokens per step cost more than that gained on our prose, tool and Chinese workloads. Only code was faster, by 2.4%.

## 7. QSA indexer split-K and chunk 4096 (Pennyroyal 3.0.1, 2026-10-12)

The `optional-qsa-indexer-splitk` set addresses CTA starvation in the indexer's prefill MQA scoring. With a 128 MiB logits budget, a 480K-deep chunk launches 8 CTAs on 188 SMs. The set splits key tiles across CTAs, choosing the split from the shapes; logits stay bit-identical.

Kernel test (`tests/test_qsa_mqa_splitk.py`, one indexer launch):

| Depth | Rows/launch | Split | Before | Split-K | Speedup | Bit-exact |
|---|---:|---:|---:|---:|---:|---|
| 8K | 2048 | 8 | 0.071 ms | 0.043 ms | 1.6× | yes |
| 32K | 2048 | 8 | 0.21 ms | 0.12 ms | 1.8× | yes |
| 128K | 992 | 16 | 0.78 ms | 0.30 ms | 2.6× | yes |
| 256K | 480 | 32 | 1.38 ms | 0.29 ms | 4.8× | yes |
| 480K | 256 | 64 | 2.48 ms | 0.29 ms | **8.5×** | yes |

Served, same production config. Arm F is now in production.

| Metric | 3.0.1 (chunk 2048) | **+ split-K (F)** | + split-K + chunk 4096 (G) |
|---|---:|---:|---:|
| Decode throughput vs 3.0.1, 24-case geomean | 1.000 | 0.994 | 1.000 |
| Cold prefill 128K / 240K | 11.9K / 10.9K tok/s | 12.1K / 11.6K | 12.8K / 12.3K |
| 490K three-needle TTFT | 66.5 s | **52.4 s** | 45.9 s |
| 480K main-agent TTFT | 74.8 s | **61.6 s** | 50.0 s |
| Longest stall of other streams during a 128K prefill (warm) | 0.22 s | 0.22 s | 0.36 s |
| Longest subagent stall during the 480K cold prefill (p99) | 0.50 s | 0.32 s (0.30) | 1.34 s (0.48) |
| Paired NLL vs 3.0.1 (8K / 32K / 128K) | — | +0.003 / +0.020 / +0.002, n.s. | n.s. |
| Minimum free VRAM during the full suite | 2749 MiB | 2649 MiB | **1001 MiB** (stress) |

- **First-use JIT.** Each new split value, and each new chunk shape, compiles a kernel on first use, which causes a one-off stall of about 2 s. Afterwards the TileLang cache in a persistent `TILELANG_CACHE_DIR` serves it.
- **Chunk 4096 is not in production.** Adding `--chunked-prefill-size 4096` on top of split-K reaches 50 s, but stress testing dropped headroom to 1.0 GiB with the 1.5M pool. A smaller pool (about 1.4M) should restore the margin; this is untested.
