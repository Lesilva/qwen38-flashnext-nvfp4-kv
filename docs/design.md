# Design

## Why the gather

In Qwen3.8 Flash-Next, QSA (the sparse full-attention layers) never reads the KV pool directly from an attention kernel. Every read goes through a gather first:

- **Decode / target-verify / draft-extend / draft-decode.** `qwen_sparse_kv_extraction_compact_triton` copies each query row's top-k (2051) selected tokens into a page-aligned scratch. FlashInfer's paged decode kernel (TRT-LLM XQA on SM120) then runs on that scratch.
- **Chunked prefill with a cached prefix.** The full context is packed with `index_select` and handed to the sparse GQA prefill kernel.
- **Prefill without a prefix** uses the freshly computed BF16 K/V and never touches the cache.

So the only change needed is in the gather: read NVFP4 from the pool and write FP8 into the same scratch the kernels already consume. The attention kernels, their CUDA graphs and the top-k selection (which reads a separate BF16 compressed index, unaffected by KV dtype) stay exactly as they are.

The vLLM community implementation unpacks FP4 inside the attention kernel and measured 1.75–2.15× slower prefill. Here prefill cost is essentially unchanged: −0.2 to −0.6 %.

## Storage format

We keep the stock SGLang `NVFP4KVCacheMethod` layout, written by `flashinfer.nvfp4_kv_quantize`:

- data: `uint8 [slots, heads, dim/2]`, packed E2M1, even element in the low nibble;
- scales: `float8_e4m3fn [slots, heads, dim/16]`, one linear (non-swizzled) scale per 16 values;
- global scale: per layer, 1.0 (checkpoint KV scales are skipped at load, same as FP8).

Value = `e2m1(nibble) × scale × global`. The kernel decodes E2M1 with an exact piecewise formula (`m<4: 0.5m; m<6: m−2; else 2m−8`), not `exp2`, so the result is bit-exact against `flashinfer.nvfp4_kv_dequantize`.

## Pieces

| File | Change |
|---|---|
| `qsa/nvfp4_kv.py` (new) | `qwen_sparse_kv_extraction_compact_nvfp4_triton` (decode top-k gather + dequant), `nvfp4_gather_dequant` (prefix gather + dequant), and an independent PyTorch `reference_dequant` |
| `qwen_sparse_attn_backend.py` | Dispatches on the pool's quant method; passes device-side global scales when writing KV (a Python float there triggers a host→device copy that breaks CUDA-graph capture); optional BF16 scratch via `SGLANG_QSA_NVFP4_SCRATCH=bf16` |
| `fp4_kv_cache_quant_method.py` | `QSANVFP4KVCacheMethod`: no full-pool FP8 dequant workspace (the generic recipe allocates 2 × slots × heads × dim bytes for FlashInfer prefill) |
| `kv_cache_configurator.py` | Swaps the QSA pool's quant method to the one above |
| `flashinfer_backend.py` | QSA models briefly construct a FlashInfer backend before replacing it; that placeholder treats the pool as unquantized so it does not demand NVFP4 FlashInfer accesses |
| `server_args.py` | With `SGLANG_QSA_NVFP4_KV=1`, skips the KV4 MHA-backend whitelist, which does not describe the QSA path |

The MTP draft layer keeps FP8 (`--speculative-draft-kv-cache-dtype fp8_e4m3`).

## Error budget

NVFP4 round-trip relative RMS error on Gaussian data is ~0.095 (FP8: ~0.027). The FP4→FP8 double rounding adds almost nothing (a BF16 scratch measured the same), so the error comes from the FP4 storage itself. End-to-end, the 24-window paired tail NLL shows no significant difference (see results.md).

Known ways to lower it further (not implemented): 4/6 adaptive block scaling on write, per-layer calibrated global scales, or FP8 K with NVFP4 V.
