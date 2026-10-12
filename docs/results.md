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
