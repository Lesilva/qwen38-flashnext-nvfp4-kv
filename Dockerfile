# Derived image: Pennyroyal 2.5.3 + QSA NVFP4 KV (+ optional fixes).
#   docker build -t qwen38-nvfp4-kv:2.5.3 .
# Remove the optional COPY lines you do not want. The overlay files are complete
# replacements of the Pennyroyal 2.5.3 sources, so the base digest is pinned.
ARG BASE=ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:81d4d3d19b21e30b895f57e3628a8dd82c02a52c17e6234f5f3ea7836fd31fb3
FROM ${BASE}

# Required: QSA NVFP4 KV cache.
COPY overlay/pennyroyal-2.5.3/nvfp4-kv/python/ /opt/pennyroyal/python/

# Optional: Mamba ping-pong slot exhaustion fix (recommended with --prefill-decode-interval).
COPY overlay/pennyroyal-2.5.3/optional-mamba-pingpong-fix/python/ /opt/pennyroyal/python/

# Optional: RecoverSSM none-mode budget fix (for --gdn-mtp-cache-mode none).
# Must come after the nvfp4-kv COPY: it replaces kv_cache_configurator.py with a
# version that contains both changes.
COPY overlay/pennyroyal-2.5.3/optional-recoverssm-budget/python/ /opt/pennyroyal/python/

ENV SGLANG_QSA_NVFP4_KV=1
