# Derived image: Pennyroyal + QSA NVFP4 KV (+ optional fixes).
#   docker build -t qwen38-nvfp4-kv:3.1.5 .                                  # Pennyroyal 3.1.5 (default)
#   docker build --build-arg PENNY=3.1.1 \
#     --build-arg BASE=ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:f2a6a8caf2d314fdcea37c2448e4548f97ab3ff201fe1e2426861347665da60b \
#     -t qwen38-nvfp4-kv:3.1.1 .                                            # Pennyroyal 3.1.1
#   docker build --build-arg PENNY=3.0.1 \
#     --build-arg BASE=ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:25400f5609a67a692a40e8567e4a0e635b619279ab411f2a363b02a06251f2d1 \
#     -t qwen38-nvfp4-kv:3.0.1 .                                            # Pennyroyal 3.0.1
#   docker build --build-arg PENNY=2.5.3 \
#     --build-arg BASE=ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:81d4d3d19b21e30b895f57e3628a8dd82c02a52c17e6234f5f3ea7836fd31fb3 \
#     -t qwen38-nvfp4-kv:2.5.3 .                                            # Pennyroyal 2.5.3
# The overlay files are complete replacements of that release's sources, so BASE and PENNY must match.
ARG BASE=ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:6a10be9d0b34a3b78d81d20c649a1278ecea8acebdec835185a2605837c44f37
FROM ${BASE}
ARG PENNY=3.1.5
USER 0:0

# Required: QSA NVFP4 KV cache.
COPY overlay/pennyroyal-${PENNY}/nvfp4-kv/python/ /opt/pennyroyal/python/

# Optional: Mamba ping-pong slot exhaustion fix (recommended with --prefill-decode-interval).
COPY overlay/pennyroyal-${PENNY}/optional-mamba-pingpong-fix/python/ /opt/pennyroyal/python/

# Optional (3.0.1, 3.1.1, 3.1.5): QSA indexer prefill split-K -- bit-identical logits, faster long-context cold prefill
# (480K main-agent TTFT -18% on one PRO 6000). Independent of NVFP4 KV; works with FP8 KV too.
COPY overlay/pennyroyal-${PENNY}/optional-qsa-indexer-splitk/python/ /opt/pennyroyal/python/

# Optional (3.1.5 only): wide-verify fast paths -- the dense FP8 linears, HC mix and FP8 lm_head keep their W8A16 /
# MIX2 fast paths up to 64 rows (NEXTN verify at 5-16 streams) and QSA decode top-k runs on many CTAs per row.
# 6-16 streams +31-50% decode on one PRO 6000. Each path also needs its env flag. Uncomment for PENNY=3.1.5:
# COPY overlay/pennyroyal-3.1.5/optional-wide-verify/python/ /opt/pennyroyal/python/
# ENV QWEN_C_MC_TOPK=1 QWEN_EP_HC_WIDE=1 QWEN_I_WIDE_LMHEAD=1 QWEN_J_WIDE_FP8LIN=1

# 2.5.3 only: RecoverSSM none-mode budget fix (3.0 ships the upstream fix). Uncomment for PENNY=2.5.3;
# it replaces kv_cache_configurator.py with a version that contains both changes.
# COPY overlay/pennyroyal-2.5.3/optional-recoverssm-budget/python/ /opt/pennyroyal/python/

ENV SGLANG_QSA_NVFP4_KV=1
