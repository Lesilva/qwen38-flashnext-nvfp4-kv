# Derived image: Pennyroyal + QSA NVFP4 KV (+ optional fixes).
#   docker build -t qwen38-nvfp4-kv:3.0.1 .                                  # Pennyroyal 3.0.1 (default)
#   docker build --build-arg PENNY=2.5.3 \
#     --build-arg BASE=ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:81d4d3d19b21e30b895f57e3628a8dd82c02a52c17e6234f5f3ea7836fd31fb3 \
#     -t qwen38-nvfp4-kv:2.5.3 .                                            # Pennyroyal 2.5.3
# The overlay files are complete replacements of that release's sources, so BASE and PENNY must match.
ARG BASE=ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:25400f5609a67a692a40e8567e4a0e635b619279ab411f2a363b02a06251f2d1
FROM ${BASE}
ARG PENNY=3.0.1
USER 0:0

# Required: QSA NVFP4 KV cache.
COPY overlay/pennyroyal-${PENNY}/nvfp4-kv/python/ /opt/pennyroyal/python/

# Optional: Mamba ping-pong slot exhaustion fix (recommended with --prefill-decode-interval).
COPY overlay/pennyroyal-${PENNY}/optional-mamba-pingpong-fix/python/ /opt/pennyroyal/python/

# 2.5.3 only: RecoverSSM none-mode budget fix (3.0 ships the upstream fix). Uncomment for PENNY=2.5.3;
# it replaces kv_cache_configurator.py with a version that contains both changes.
# COPY overlay/pennyroyal-2.5.3/optional-recoverssm-budget/python/ /opt/pennyroyal/python/

ENV SGLANG_QSA_NVFP4_KV=1
