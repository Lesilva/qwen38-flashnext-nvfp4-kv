#!/usr/bin/env bash
# Print read-only `-v` mounts for the overlay, to paste into your own `docker run`
# on top of the stock Pennyroyal 2.5.3 image (no image build needed). Usage:
#   ./examples/overlay-mounts.sh [--with-pingpong-fix] [--with-recoverssm-budget]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)/overlay/pennyroyal-2.5.3"
PINGPONG=0; BUDGET=0
for arg in "$@"; do
  case "$arg" in
    --with-pingpong-fix) PINGPONG=1 ;;
    --with-recoverssm-budget) BUDGET=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
emit() {  # emit <set>: one mount per file in that set
  find "$ROOT/$1" -type f -name '*.py' | sort | while IFS= read -r f; do
    rel="${f#"$ROOT/$1/"}"
    # recoverssm-budget ships a kv_cache_configurator.py that already contains the
    # nvfp4-kv change; when selected, it replaces the nvfp4-kv copy.
    if [ "$1" = nvfp4-kv ] && [ "$BUDGET" = 1 ] && [ "$rel" = python/sglang/srt/mem_cache/kv_cache_configurator.py ]; then
      continue
    fi
    printf -- '-v %q:/opt/pennyroyal/%s:ro \\\n' "$f" "$rel"
  done
}
emit nvfp4-kv
[ "$PINGPONG" = 1 ] && emit optional-mamba-pingpong-fix
[ "$BUDGET" = 1 ] && emit optional-recoverssm-budget
echo "-e SGLANG_QSA_NVFP4_KV=1 \\"
