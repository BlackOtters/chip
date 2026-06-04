#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 2 ]]; then
  echo "Usage: ./scripts/apply_overlay.sh /path/to/mjlab /path/to/RoboJuDo" >&2
  exit 1
fi

MJLAB_TARGET="$(cd "$1" && pwd)"
ROBOJUDO_TARGET="$(cd "$2" && pwd)"

copy_tree() {
  local src_dir="$1"
  local dst_dir="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src_dir/" "$dst_dir/"
  else
    cp -a "$src_dir/." "$dst_dir/"
  fi
}

copy_tree "$ROOT_DIR/train/mjlab_overlay" "$MJLAB_TARGET"
copy_tree "$ROOT_DIR/deploy/robojudo_overlay" "$ROBOJUDO_TARGET"

echo "Overlay applied."
echo "mjlab target: $MJLAB_TARGET"
echo "RoboJuDo target: $ROBOJUDO_TARGET"
