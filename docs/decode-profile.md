# Where a decode step goes (Pennyroyal 3.1.5, 1× RTX PRO 6000)

Measured 2026-11-14 on my production configuration: OrcaRouter NVFP4 weights, NVFP4 KV, 524K context, NEXTN 3 steps / 4 draft tokens with a 96K FR-Spec map, CUDA graphs on, 350 W cap. Measured with a timing hook in the speculative worker and `torch.profiler`.

## Summary

- One single-stream decode round takes about 10 ms and yields about 2.86 tokens.
- The GPU is executing kernels 97% of that time. The scheduler thread issues the whole round in 1.16 ms and then waits, so host-side orchestration has nothing to give back.
- Target verification is about 85% of the round; the draft head (draft + extend) is about 15%.
- About two thirds of the round is kernels that read weights, and those already run at about 1.6 TB/s, close to what the card's memory can deliver. Kernel fusion cannot shorten them.
- Kernels on parallel streams share that bandwidth. Moving a weight-reading kernel "off the critical path" onto another stream does not make the round shorter.

## Per phase (synchronised timing, medians)

| Case | draft | verify | extend | outside |
|---|---:|---:|---:|---:|
| 1 stream, short context | 0.96 ms | 9.01 ms | 0.72 ms | 0.49 ms |
| 1 stream, 250K context | 1.04 ms | 10.07 ms | 0.85 ms | 0.53 ms |
| 2 streams | 1.06 ms | 11.31 ms | 0.79 ms | 0.50 ms |
| 4 streams | 1.12 ms | 15.51 ms | 0.88 ms | 0.54 ms |

Synchronising adds about 1.3 ms per round. Unsynchronised periods are 9.94 ms (short) and 10.70 ms (250K).

## Kernel time by family (1 stream, graph replay)

About 2,200 kernels per round. Shares are of summed kernel time (12.67 ms; parallel branches overlap, the busy union is 10.0 ms).

| Family | Per round | Share |
|---|---:|---:|
| Expert MoE (NVFP4 grouped GEMM and routing) | 3.77 ms | 29.7% |
| W8A16 dense GEMV | 3.63 ms | 28.6% |
| BF16 GEMM | 1.57 ms | 12.4% |
| Hyper-connection kernels | 1.27 ms | 10.0% |
| Full attention, QSA, KV | 0.81 ms | 6.4% |
| Copies, indexing, fills | 0.71 ms | 5.6% |
| GDN linear attention | 0.55 ms | 4.4% |
| Other | 0.26 ms | 2.0% |

## One linear-attention layer (36 of 48 layers, about 164 µs each)

| Step | Measured | Weight bytes read | Floor at 1.6 TB/s |
|---|---:|---:|---:|
| Hyper-connection mix | 9.3 µs | 6.6 MB | ~4 µs |
| `in_proj_qkvz` GEMV, beside `in_proj_ba` | 33.5 µs | 41.9 MB | 26 µs |
| GDN middle (split, conv, copies, decode) | 17.5 µs | small | ~3–5 µs |
| `out_proj` GEMV | 13.4 µs | 15.7 MB | 9.8 µs |
| Hyper-connection combine + mix | 12 µs | 6.6 MB | ~4 µs |
| MoE router and maps | ~12.5 µs | 2.6 MB | ~2–4 µs |
| MoE up/gate grouped GEMM | 39–48 µs | ~74 MB (40 experts) | 46 µs |
| MoE down grouped GEMM | 23–28 µs | ~37 MB | 23 µs |

A 4-row verify reads 40 experts per layer (4 rows × top-10). That, not the kernel, is the MoE cost. The LM head reads 635 MB in 394 µs per round.

## 250K context

Only two kernels grow: the QSA index top-k (0.016 → 0.517 ms per round) and the QSA scoring kernel (0.066 → 0.296 ms).

## What I tried

I replaced the expert MoE with a single-kernel implementation and two small BF16 GEMMs with a custom GEMV, loaded out of tree. Result: −4.1% step period at 1 stream and nothing at 2+ streams (benchmark geomean 1.048 at 1 stream, 1.001 at 2–16 streams). The GEMV did not help because of the shared bandwidth above. That is below my adoption bar, so it is not in production and not in this repository.

## Limits

- One GPU, one checkpoint, one configuration.
- The bandwidth floor uses 1.6 TB/s, taken from the LM-head GEMV; I checked the large kernels against it, not every kernel.
- With graphs off the round takes 52 ms; module-level attribution comes from that mode and is used only for shares.
