# qwen38-flashnext-nvfp4-kv

**NVFP4 KV cache for Qwen3.8 Flash-Next's QSA sparse attention on SGLang (Pennyroyal build).**

It cuts KV memory by ~41% at ~1–2% decode cost. On a single RTX PRO 6000 (96 GB) that is enough for a 1.5M-token pool, 512K context and prefill/decode interleaving all at once. The quality difference was not measurable (24-window paired NLL, 8K–128K).

[中文说明见下方](#中文说明)

| | FP8 KV (stock) | NVFP4 KV (this repo) |
|---|---:|---:|
| Target KV pool, 1M tokens | 12.16 GiB | **7.15 GiB** |
| Free VRAM after CUDA graphs (1M pool) | ~2.3 GiB | **~7.35 GiB** |
| Decode throughput (24-case geomean, same config) | 1.000 | 0.989 |
| MTP acceptance length | 2.297 | 2.315 |
| Cold prefill 8K–240K | 11.1–9.7K tok/s | −0.2…−0.6 % |
| 490K three-needle retrieval | pass | pass |

Full numbers: [docs/results.md](docs/results.md). How it works: [docs/design.md](docs/design.md).

## Status

- **Pennyroyal 3.1.1** (`ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:f2a6a8ca…`): in production since 2026-10-22, full suite passed. The rebase was a clean three-way merge with our hunks unchanged; decode, prefill, quality, long-context and headroom match 3.0.1 (decode geomean 1.002). We keep Penny's NEXTN MTP profile, which Penny 3.1 marks deprecated in favour of DFlash2 (see Limitations).
- **Pennyroyal 3.0.1** (`…@sha256:25400f56…`): ran in production 2026-10-12 → 10-22, full suite passed. Relative to our 2.5.3 deployment, it is +5.6% decode (geomean; +13–18% at 1–2 streams) and +12–13% cold prefill.
- **Pennyroyal 2.5.3** (`…@sha256:81d4d3d1…`): validated and ran in production 09-28 → 10-12. Earlier, the same code ran on 2.5.2.
- All three releases are on the SGLang 0.5.19 integration base.
- Hardware: **1× RTX PRO 6000 Blackwell 96 GB (SM120)**, 350 W cap. Other SM100/SM120 GPUs should work but are untested.
- Checkpoint used: `jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4` (NVFP4 weights). Any Qwen3.8 Flash-Next checkpoint that Pennyroyal serves through its QSA backend should work; others are untested.
- Not upstream. Upstream SGLang has no QSA NVFP4 KV path yet (as of 2026-10-12). vLLM has an open PR for Qwen4Exp (#54846). See [Related work](#related-work) for community implementations.

## What's in here

| Patch set | Required? | Files | What it does |
|---|---|---|---|
| `nvfp4-kv` | **required** | `qsa/nvfp4_kv.py` (new), `qwen_sparse_attn_backend.py`, `fp4_kv_cache_quant_method.py`, `kv_cache_configurator.py`, `flashinfer_backend.py`, `server_args.py` | Stores the QSA layers' KV as NVFP4 (packed E2M1 + one E4M3 scale per 16 values). QSA's own top-k and prefix gathers dequantize to FP8 on the fly, so the attention kernels are unchanged and no full-pool dequant workspace is allocated. |
| `optional-mamba-pingpong-fix` | optional, recommended with `--prefill-decode-interval` | `managers/scheduler.py`, `mem_cache/unified_radix_cache.py` | Fixes a scheduler crash (`Not enough space for mamba ping pong` / `Can not alloc mamba cache`) under mixed long/short agent load. (A) releases the COW-source lock on the early-return path of `cache_unfinished_req`. This part only takes effect with `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=true`. (B) adds a Mamba-slot admission check before batching. Upstream: sgl-project/sglang#34676, #39787. |
| `optional-qsa-indexer-splitk` | optional (3.0.1), **independent of NVFP4 KV** | `layers/attention/qsa/mqa.py` | Fixes CTA starvation in the QSA indexer's prefill MQA scoring. At long context the 128 MiB logits budget leaves ~8 CTAs per launch on 188 SMs. Key tiles are now split across CTAs, with the split chosen from the shapes (≈512 CTAs, ≥4 tiles each, no device sync). Logits are bit-identical (each one is still computed by exactly one CTA). Single-launch speedup is 1.6× at 8K and 8.5× at 480K. Served: 490K needle TTFT 66.5 → 52.3 s, 480K main-agent TTFT 74.8 → 61.6 s, decode unchanged. `SGLANG_QSA_MQA_SPLITK=1` disables it; an integer forces a split. The first use of each split value JIT-compiles a TileLang kernel (a one-off ~2 s stall). Keep `TILELANG_CACHE_DIR` on a persistent mount, or prewarm with `tests/test_qsa_mqa_splitk.py`. Idea credit: redashes1984 (fixed split=64). |
| `optional-recoverssm-budget` | **2.5.3 only**, needed there for `--gdn-mtp-cache-mode none` | `mem_cache/kv_cache_configurator.py` | Correct memory budgeting for conv windows in RecoverSSM none mode. This file also contains the `nvfp4-kv` change and replaces that set's copy. Pennyroyal 3.0 ships an upstream fix (Penny #51), so 3.0.1 has no such set. |

Each set comes in two forms:
- `overlay/pennyroyal-<ver>/<set>/python/...`: complete replacement files, for bind mounts or `COPY`.
- `patches/pennyroyal-<ver>/000N-*.patch`: unified diffs against that release's source root. Apply with `git apply` or `patch -p1` in numeric order; 0001 and 0002 also apply on their own.

Available versions: `pennyroyal-3.1.1` and `pennyroyal-3.0.1` (0001 NVFP4 KV, 0002 ping-pong fix, 0003 indexer split-K) and `pennyroyal-2.5.3` (0001, 0002, 0003 RecoverSSM budget).

## Quick start

**A. Build a derived image** (edit the optional `COPY` lines first):

```bash
git clone https://github.com/Lesilva/qwen38-flashnext-nvfp4-kv
cd qwen38-flashnext-nvfp4-kv
docker build -t qwen38-nvfp4-kv:3.1.1 .      # see the Dockerfile header for 3.0.1 / 2.5.3
MODEL_DIR=/path/to/checkpoint ./examples/run-tested-config.sh
```

**B. No build: bind-mount the overlay** into the stock Pennyroyal image of the same version (run it as `--user 0:0` or adjust ownership; 3.0 images default to user `penny`):

```bash
./examples/overlay-mounts.sh --with-pingpong-fix --with-splitk                     # 3.1.1
./examples/overlay-mounts.sh --penny 3.0.1 --with-pingpong-fix --with-splitk
./examples/overlay-mounts.sh --penny 2.5.3 --with-pingpong-fix --with-recoverssm-budget
# paste the printed -v/-e lines into your docker run
```

Then add these flags to your launch:

```
--kv-cache-dtype nvfp4
--speculative-draft-kv-cache-dtype fp8_e4m3   # MTP draft layer stays FP8
--max-total-tokens <N>                        # required, see below
```

Keep `SGLANG_QSA_NVFP4_KV=1` in the environment. Without it, the stock code rejects `nvfp4` for QSA models. To switch back to FP8, set `--kv-cache-dtype fp8_e4m3`; nothing else changes.

**`--max-total-tokens` is required.** The stock capacity planner still budgets a full-pool FP8 dequant workspace (and sizes the draft pool from the target dtype), and this path allocates neither. With our 71 GiB-weight checkpoint on 96 GB, `1500000` leaves ~2 GiB free under stress. Start lower on other checkpoints and raise it while watching `nvidia-smi`.

## Verify

Run the unit tests on the GPU; they take about a minute:

```bash
docker run --rm --gpus device=0 --entrypoint /opt/pennyroyal/.venv/bin/python \
  -e PYTHONPATH=/opt/pennyroyal/python -v "$PWD/tests:/tests" \
  qwen38-nvfp4-kv:3.0.1 /tests/test_nvfp4_kv.py /tmp/nvfp4-unit.json
```

What the tests check:
- **T1** Layout probe. FlashInfer quantize, then FlashInfer dequantize, must match our independent PyTorch reference bit for bit. Relative error is 0.095 with the correct layout and 1.41 with swapped nibbles.
- **T3** Decode top-k gather: bit-exact, including `-1` / out-of-range slots and `topk=2051`.
- **T4** Prefix gather: bit-exact.
- **T5** Timings.
- `tests/test_qsa_mqa_splitk.py` (split-K set): bit-exact split vs no split at 8K–480K depth and on a packed batch, a check against the torch reference, and timing per depth. Running it with `/cache` mounted and `TILELANG_CACHE_DIR=/cache/tilelang` also pre-compiles the kernels.

Server log at startup should include `KV4: QSA NVFP4 KV enabled via SGLANG_QSA_NVFP4_KV=1`, and `/get_server_info` should report `"kv_cache_dtype": "nvfp4"`.

## Limitations

- Served and benchmarked with **all sets for that version together**. The sets touch independent code paths, but `nvfp4-kv` alone has not been run as a server yet.
- Pinned per release. The overlay files replace whole source files, so only mount them into the matching Pennyroyal version. Rebased sets are added per release as `overlay/pennyroyal-<ver>/`.
- Capacity planning is not fixed (see `--max-total-tokens` above).
- NVFP4 is storage-only. Attention still runs on an FP8 scratch buffer (`SGLANG_QSA_NVFP4_SCRATCH=bf16` switches it to BF16; we measured no quality difference).
- Checkpoint KV scales are not loaded (global scale = 1.0, same as the stock FP8 path).
- Tested with `--mamba-radix-cache-strategy extra_buffer_lazy`, MTP 3 steps, page size 64. On 3.0.1 we also served 3.0's adaptive MTP (4/8-token width at C1) with this patch set: it passed the full suite, but on our workload it was 3% slower at C1 than fixed MTP and used 0.86 GiB more VRAM, so production stays on fixed MTP. Other combinations are untested.
- **DFlash2 (Penny 3.1's new default) is not validated with this patch set.** On 3.1.1 we served Penny's DFlash2 speculative settings on top of these sets (NVFP4 KV, `extra_buffer_lazy`, 16 streams, 1.5M pool). C1 decode was 12% slower than NEXTN, 480K cold TTFT went from 61 s to 173 s, the concurrency test failed, and resuming from decode-time Mamba checkpoints produced wrong logprobs (#17 probe 3.99 vs 0.06). Penny's own DFlash2 profile uses FP8 KV and `extra_buffer`, which we have not tested with these sets. Use NEXTN with these patches.

## Related work

We wrote this independently. Afterwards we found the same core idea (dequantize NVFP4 inside QSA's gather, with no full-pool workspace) already public in SGLang-based stacks:
- [Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) ("dspark", MiaAI Lab): the original SGLang patch, on 2× DGX Spark (SM121).
- [redashes1984/sglang-flashnext-nvfp4kv-sm120](https://github.com/redashes1984/sglang-flashnext-nvfp4kv-sm120): a port of the dspark patch to the RTX PRO 6000, plus an expert cold pool and a 1M-context deployment.

Differences here:
- We dequantize into an **FP8** scratch, so the stock FP8 XQA decode kernel is reused and the gather is faster than the plain FP8 copy. Those projects dequantize to BF16.
- We ship bit-exact kernel tests against FlashInfer and a paired 24-window NLL measurement.
- We provide overlays rebased onto each Pennyroyal release.
- redashes1984 also published the indexer split-K diagnosis and a fixed split=64 patch. Our split-K set is an independent implementation on Pennyroyal 3.0.1 that picks the split adaptively per launch, with bit-exactness tests.

## License

Apache-2.0, same as SGLang and Pennyroyal. The overlay files are modified copies of SGLang/Pennyroyal sources and keep their original notices. See [NOTICE](NOTICE).

---

## 中文说明

这个仓库给 **Qwen3.8 Flash-Next 的 QSA 稀疏注意力**加上 **NVFP4 KV cache**，基于 SGLang 的 Pennyroyal 版本（SM120 发行版）。

- KV 显存减少约 41%，解码只慢约 1～2%。
- 在单张 RTX PRO 6000（96 GB）上，可以同时做到：150 万 token 的 KV 池、512K 上下文、长 prefill 期间其他流不卡顿。
- 精度：用 24 个窗口做配对的尾部 NLL 比较，8K～128K 都没有测出差异。

**做法**：QSA 每个 query 只看 top-k 的 2051 个 token，读 KV 之前本来就有一步 gather。我们只改这一步：gather 的同时把 NVFP4 反量化成 FP8，注意力 kernel 完全不动，也不分配整池大小的反量化 workspace。详见 [docs/design.md](docs/design.md)。

**内容**：
- 必需：`nvfp4-kv`。
- 可选：`optional-mamba-pingpong-fix`，修复长短 agent 混跑时调度器因 Mamba 状态槽耗尽而崩溃的问题。开 `--prefill-decode-interval` 时建议装。
- 可选（仅 2.5.3）：`optional-recoverssm-budget`，修正 `--gdn-mtp-cache-mode none` 下的显存预算。Pennyroyal 3.0 已自带上游修复，不再需要。

每组都提供两种形式：完整文件（overlay，挂载或 COPY 即可）和 diff 补丁（patches，可以 `git apply`）。

**用法**：
1. 用 Dockerfile 构建派生镜像，或者用 `examples/overlay-mounts.sh` 生成只读挂载参数。
2. 启动时加上：
   - `--kv-cache-dtype nvfp4`
   - `--speculative-draft-kv-cache-dtype fp8_e4m3`
   - `--max-total-tokens <N>`
   - 环境变量 `SGLANG_QSA_NVFP4_KV=1`
3. 想换回 FP8，只需把参数改成 `--kv-cache-dtype fp8_e4m3`。

**注意**：
- 完整的服务测试是在三组补丁一起装的情况下做的；只装 `nvfp4-kv` 的组合还没有单独起服务测过。
- `--max-total-tokens` 必须手动指定，因为容量规划还没有修正。
- 已在 Pennyroyal **3.1.1**（10-22 起用于生产，补丁三方合并无冲突，速度、精度和 3.0.1 持平）、3.0.1 和 2.5.3 上验证过（更早也在 2.5.2 上跑过生产），硬件是 RTX PRO 6000。3.0.1 相对 2.5.3：解码快约 5.6%，冷 prefill 快 12～13%。
- 3.0 的自适应 MTP 和本补丁可以一起用，但在我们的负载下单路慢 3%，所以生产仍用固定 3 步。
- 社区里已有思路相同的实现（dspark、redashes1984），见上方 Related work。
- 可选（3.0.1、3.1.1）：`optional-qsa-indexer-splitk`，修复 QSA indexer 在长上下文 prefill 时 CTA 太少的问题，结果逐位一致，480K 冷启动首字时间缩短 18%。它和 NVFP4 无关，FP8 KV 也能用。思路来自 redashes1984，我们改成了按形状自动选择拆分数。
- Penny 3.1 默认改用 DFlash2，并把原生 MTP 标为弃用。我们在 3.1.1 上把 DFlash2 和这套补丁一起测过：单路慢 12%、480K 冷启动从 61 秒变成 173 秒、并发测试失败，而且从 Mamba 解码检查点续算的结果是错的。所以配合本补丁请继续用 NEXTN MTP。
- overlay 是整文件替换，不要挂到其他版本上。

完整测试数据见 [docs/results.md](docs/results.md)。
