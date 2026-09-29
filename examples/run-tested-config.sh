#!/usr/bin/env bash
# The exact serving configuration this patch set was validated with
# (1x RTX PRO 6000 Blackwell 96 GB, 350 W, 89 GiB host RAM).
# Adjust MODEL_DIR and anything marked "checkpoint-specific" for your setup.
set -euo pipefail
IMAGE="${IMAGE:-qwen38-nvfp4-kv:2.5.3}"      # built from ../Dockerfile
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the checkpoint directory}"
PORT="${PORT:-30000}"

docker run -d --name qwen38-nvfp4-kv \
  --gpus device=0 --network host --ipc host --ulimit memlock=-1:-1 \
  --memory 78g \
  -e SGLANG_QSA_NVFP4_KV=1 \
  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=true \
  -e SGLANG_MAMBA_CONV_DTYPE=bfloat16 \
  -e SGLANG_SM120_ONLINE_MXFP8=true \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$MODEL_DIR":/models/nvfp4:ro \
  --entrypoint /opt/pennyroyal/.venv/bin/python \
  "$IMAGE" -m sglang.launch_server \
  --model-path /models/nvfp4 --host 0.0.0.0 --port "$PORT" \
  --quantization modelopt_fp4 --dtype bfloat16 \
  --kv-cache-dtype nvfp4 \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --max-total-tokens 1500000 \
  --mem-fraction-static 0.98 \
  --context-length 524288 \
  --json-model-override-args '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}' \
  --page-size 64 --max-running-requests 16 \
  --chunked-prefill-size 2048 --prefill-decode-interval 1 \
  --mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 \
  --max-mamba-cache-size 48 --mamba-max-states-per-path 5 --mamba-track-interval 64 \
  --gdn-mtp-cache-mode none \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --ple-offload-embedding \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant \
  --cuda-graph-bs-decode 1 2 3 4 5 6 8 12 16 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
# Notes
# - --kv-cache-dtype nvfp4 + SGLANG_QSA_NVFP4_KV=1 enable the QSA NVFP4 path. Switch back
#   to FP8 with --kv-cache-dtype fp8_e4m3 (same image, env var ignored).
# - --max-total-tokens is REQUIRED with NVFP4 for now: the stock capacity planner still
#   budgets a full-pool dequant workspace that this path never allocates. 1,500,000 left
#   ~2 GiB free under load with the 71 GiB-weight checkpoint we used; start lower on
#   other checkpoints and raise it while watching nvidia-smi.
# - --prefill-decode-interval 1 keeps other streams decoding during long prefills;
#   use it together with the optional ping-pong fix.
# - --gdn-mtp-cache-mode none needs the optional RecoverSSM budget fix.
# - --ple-offload-embedding, --quantization modelopt_fp4, the online MXFP8 flag and the
#   YaRN override are checkpoint-/goal-specific (we serve 512K via YaRN x2).
