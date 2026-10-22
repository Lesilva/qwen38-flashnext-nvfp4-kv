#!/usr/bin/env bash
# Print read-only `-v` mounts for the overlay, to paste into your own `docker run`
# on top of the stock Pennyroyal image of the same version (no image build needed). Usage:
#   ./examples/overlay-mounts.sh [--penny 3.1.1|3.0.1|2.5.3] [--with-pingpong-fix] [--with-splitk] [--with-recoverssm-budget]
# --with-recoverssm-budget exists for 2.5.3 only; 3.0 ships the upstream fix.
set -euo pipefail
PENNY=3.1.1; PINGPONG=0; BUDGET=0; SPLITK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --penny) PENNY="$2"; shift ;;
    --with-pingpong-fix) PINGPONG=1 ;;
    --with-splitk) SPLITK=1 ;;
    --with-recoverssm-budget) BUDGET=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)/overlay/pennyroyal-$PENNY"
[ -d "$ROOT" ] || { echo "no overlay for Pennyroyal $PENNY" >&2; exit 2; }
if [ "$SPLITK" = 1 ] && [ ! -d "$ROOT/optional-qsa-indexer-splitk" ]; then
  echo "optional-qsa-indexer-splitk is only shipped for Pennyroyal 3.0.1 and 3.1.1" >&2; exit 2
fi
if [ "$BUDGET" = 1 ] && [ ! -d "$ROOT/optional-recoverssm-budget" ]; then
  echo "optional-recoverssm-budget is not needed (and not shipped) for Pennyroyal $PENNY" >&2; exit 2
fi
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
[ "$SPLITK" = 1 ] && emit optional-qsa-indexer-splitk
[ "$BUDGET" = 1 ] && emit optional-recoverssm-budget
echo "-e SGLANG_QSA_NVFP4_KV=1 \\"
