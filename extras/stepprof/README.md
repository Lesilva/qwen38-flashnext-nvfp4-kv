# Decode step profiler

A measurement hook for one speculative decode round on Pennyroyal 3.1.5 (`EAGLEWorkerV2`: draft, target verify, draft extend). It is inert unless `STEPPROF_DIR` is set.

- `pennyroyal-3.1.5-step-profile-hook.patch`: apply to the source root, or mount the patched `eagle_worker_v2.py`.
- Driven by `$STEPPROF_DIR/ctl.json`: `{"id": "run1", "mode": "nosync" | "wall" | "prof", "steps": 300, "skip": 20, "bs": 1}`.
  - `nosync`: timestamps at phase boundaries, no synchronisation. The median gap between rounds is the true step period with graphs on.
  - `wall`: `torch.cuda.synchronize()` before each timestamp, for per-phase wall time (adds about 1.3 ms per round).
  - `prof`: `torch.profiler` over N rounds with one range per phase; with `STEPPROF_MODULES=1` and `--disable-cuda-graph`, one range per `nn.Module` call.
- Results land in `$STEPPROF_DIR/<id>.json` (plus `.trace.json` and `.table.txt` for `prof`).
- `analyze.py <dir>`: phase medians, kernel time by phase, module and kernel.
- `kernel_sequence.py <trace> <out> <steps>`: ordered kernel timeline of one round.
- `glue_vs_weights.py <out>`: wall time covered by weight-reading kernels versus everything else.

Check which worker your configuration uses before patching: NEXTN with `enable_multi_layer_eagle=False` resolves to `EAGLEWorkerV2`.
