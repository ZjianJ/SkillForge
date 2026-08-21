#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${OFFICIAL_BASELINES_ROOT:-$ROOT/third_party/official}"

clone_pinned() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local path="$DEST/$name"

  if [[ ! -d "$path/.git" ]]; then
    git clone --filter=blob:none "$url" "$path"
  fi
  git -C "$path" fetch --depth 1 origin "$commit"
  git -C "$path" checkout --detach "$commit"
  local actual
  actual="$(git -C "$path" rev-parse HEAD)"
  if [[ "$actual" != "$commit" ]]; then
    echo "commit verification failed for $name: $actual != $commit" >&2
    exit 1
  fi
  echo "$name $actual"
}

mkdir -p "$DEST"
clone_pinned \
  SE-KD3x \
  https://github.com/almogtavor/SE-KD3x.git \
  08b276383a31fe5c07eb6685f9c4557b78e42880
clone_pinned \
  LMOps \
  https://github.com/microsoft/LMOps.git \
  4f2a9deb5f08e459fd44c2e4792344d78ca89fc3

echo "Official baselines are ready in $DEST"
