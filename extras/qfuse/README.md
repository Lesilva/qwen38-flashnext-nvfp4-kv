# qfuse (experimental)

Out-of-tree fused decode kernels for Qwen3.8 Flash-Next on Pennyroyal 3.1.5 (RTX PRO 6000, SM120), plus the loader that lets them survive Pennyroyal upgrades without file merges.

**Status: works and is correct, but I do not run it in production.** With the one kernel that helps (`moe`), single-stream decode is about 4–5% faster and 2+ streams are unchanged, which is below my adoption bar. It is here because the loader pattern and the measurements are reusable.

## Measured (2026-11-14, same-window A/B, 350 W cap)

| | Stock 3.1.5 | qfuse `moe` |
|---|---:|---:|
| Decode step period, 1 stream, short context | 9.96 ms | 9.56 ms (−4.1%) |
| Decode step period, 1 stream, 250K context | 10.69 ms | 10.34 ms (−3.3%) |
| Decode step period, 2 streams | 12.30 ms | 12.18 ms (−1.0%) |
| Decode step period, 4 streams | 16.16 ms | 16.29 ms (+0.8%) |
| Benchmark, 1 stream, geomean of 4 workloads | 1.000 | 1.048 |
| Benchmark, 2–16 streams, geomean of 28 cases | 1.000 | 1.001 |
| Benchmark, worst single case | | 0.953 |

- **Correctness**: `moe` differs from the FlashInfer CUTLASS runner by a relative L2 of 0.0045 (same W4A4 recipe, different rounding), deterministic across runs. `ba` and `gate` match the stock result to rounding (max relative L2 6e-4).
- **`ba` and `gate` do not help.** The 33 µs BF16 GEMM for `in_proj_ba` looked like an easy target, but it is slow only because it shares memory bandwidth with the `in_proj_qkvz` GEMV running on the other stream. My kernel takes 5.7 µs alone and 34 µs in place, and it slows the GEMV next to it. See [docs/decode-profile.md](../../docs/decode-profile.md).

## Kernels

| Name | Default | What it replaces | Source |
|---|---|---|---|
| `moe` | on | FlashInfer CUTLASS W4A4 expert MoE for 1–8 rows | `csrc/qf_moe.cu`, from [SSHdotCodes/qwen-3.8-flash-next-pro6000](https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000) @ b19f075 (Apache-2.0), with the persistent scratch moved to an explicit pre-allocation |
| `ba` | off | BF16 GEMM of `linear_attn.in_proj_ba` | `csrc/qf_gemv.cu` |
| `gate` | off | BF16 GEMM of the MoE router `mlp.gate` | `csrc/qf_gemv.cu` |

## How it loads

Nothing in the image is replaced on disk.

1. `boot/` goes on `PYTHONPATH`. Its `sitecustomize.py` does nothing unless `QFUSE` is `on` or `selftest`.
2. A post-import hook wraps a few functions once their module is imported: `_run_flashinfer_cutlass`, `Qwen3_5GatedDeltaNet._forward_input_proj`, `Qwen2MoeSparseMoeBlock._forward_router_experts`.
3. Each wrapper takes the fused path only for decode-size calls (1–8 rows, expected dtypes and shapes) and calls the original otherwise. Prefill and 3+ streams are untouched.
4. Every layer is compared once against the original result, outside CUDA-graph capture, before it turns fast. A layer that fails the comparison stays on the original path.

## Upgrade contract

Each wrapper is tied to a SHA-256 of the source it replaces (`qfuse/contracts.json`, with the source text in `qfuse/snapshots/`). At start-up a mismatch leaves that kernel off and serving is stock, so an upgrade cannot break serving; the worst case is no speed-up.

On a new Pennyroyal image:

```bash
docker run --rm -v $PWD/extras/qfuse:/opt/qfuse --entrypoint /opt/pennyroyal/.venv/bin/python <image> /opt/qfuse/csrc/build.py
```

```bash
docker run --rm -v $PWD/extras/qfuse:/opt/qfuse --entrypoint /opt/pennyroyal/.venv/bin/python <image> /opt/qfuse/tools/check.py
```

`check.py` prints `OK` or `CHANGED` per target, with a diff against the snapshot. If everything is `OK`, nothing needs to change. After reviewing a change, `check.py --update` rewrites the contract.

## Run

```bash
docker run ... \
  -v $PWD/extras/qfuse:/opt/qfuse:ro -v /some/writable/dir:/qfuse-out \
  -e PYTHONPATH=/opt/qfuse/boot -e QFUSE=on \
  <image> ...
```

| Variable | Values |
|---|---|
| `QFUSE` | `off` (default), `on`, `selftest` (always return the original result; run the fused path beside it outside graph capture and record the error; use with `--disable-cuda-graph`) |
| `QFUSE_KERNELS` | comma list of `moe`, `ba`, `gate`; default `moe` |
| `QFUSE_OUT` | directory for `qfuse.log` and `state-<pid>.json` (counters, contract results, selftest statistics) |
| `QFUSE_STRICT` | `1` raises on a contract mismatch instead of disabling the kernel |

`tools/gpu_unit.py` is a few-second GPU test of the GEMV kernel against an exact reference.

## Limits

- Built and tested only on Pennyroyal 3.1.5, SM120, with the OrcaRouter NVFP4 checkpoint (hidden 2560, 512 experts, top-10).
- `moe` keeps one shared scratch buffer, so it assumes decode batches are launched one at a time, which is how the scheduler runs them.
- `moe` changes logits at rounding level. I did not measure KL divergence for it separately.
