#!/usr/bin/env bash
# setup_repo.sh -- 下載並核對 (1) 官方 4DSloMo 程式（commit 9f75dd56f34a3440d3333bc048b9e2d5b4497cf3，直接從官方 GitHub 下載）
#                  (2) 本實驗資料 data_slomo_k12_T.tar（本 repo 的 GitHub Release v1），套 code/patches，把工具放進官方 repo。
# usage: bash setup_repo.sh        可選環境變數：BIG=<放大檔的資料夾，預設 ./big>  WORK=<工作資料夾，預設 ./work>  LIMIT=<下載限速，預設 20m>
# 可重複執行：已下載/已解壓/已套 patch 的步驟會跳過。
set -euo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIG=${BIG:-$BUNDLE/big}; WORK=${WORK:-$BUNDLE/work}; LIMIT=${LIMIT:-20m}
SHA=9f75dd56f34a3440d3333bc048b9e2d5b4497cf3
CODE_URL=https://github.com/OpenImagingLab/4DSloMo/archive/$SHA.tar.gz
DATA_URL=https://github.com/Arthur031221/4dslomo-n3dv-k12/releases/download/v1/data_slomo_k12_T.tar
DATA_MD5=f1797cf12c4d025ae85a9fa0f9f4c678
mkdir -p "$BIG" "$WORK"
get() {   # get URL OUT : one file, HTTPS only, rate limited, resumable (.part)
  if [ -s "$2" ]; then echo "present: $2"; return 0; fi
  echo "download: $1"
  curl -fL --proto '=https' --tlsv1.2 --limit-rate "$LIMIT" --retry 5 --retry-delay 10 -C - -o "$2.part" "$1"
  mv "$2.part" "$2"
}
# (2) data
get "$DATA_URL" "$BIG/data_slomo_k12_T.tar"
( cd "$BIG" && echo "$DATA_MD5  data_slomo_k12_T.tar" | md5sum -c - ) || { echo "DATA MD5 MISMATCH: delete $BIG/data_slomo_k12_T.tar and rerun"; exit 1; }
# (1) official code
REPO="$WORK/4DSloMo-$SHA"
if [ ! -d "$REPO" ]; then
  get "$CODE_URL" "$BIG/4DSloMo-9f75dd56.tar.gz"
  tar -xzf "$BIG/4DSloMo-9f75dd56.tar.gz" -C "$WORK"
fi
if [ ! -e "$REPO/.bundle_patched" ]; then
  # file-by-file md5 of the official code (independent of how GitHub compresses the archive)
  ( cd "$WORK" && md5sum --quiet -c "$BUNDLE/code/OFFICIAL_CODE_MANIFEST.md5" ) || { echo "OFFICIAL CODE MISMATCH vs code/OFFICIAL_CODE_MANIFEST.md5 -- stop and report"; exit 1; }
  echo "official code: $(wc -l < "$BUNDLE/code/OFFICIAL_CODE_MANIFEST.md5") files match the manifest"
fi
[ -e "$WORK/4DSloMo" ] || ln -s "$REPO" "$WORK/4DSloMo"
cd "$REPO"
if [ ! -e .bundle_patched ]; then
  for p in "$BUNDLE"/code/patches/*.diff; do patch -p1 --dry-run < "$p" >/dev/null && patch -p1 < "$p"; done
  date > .bundle_patched
fi
cp "$BUNDLE/tools/render_s3_v2.py" "$BUNDLE/tools/render_views.py" "$REPO/"      # they import the repo packages (scene, gaussian_renderer, ...), so they live in the repo root
mkdir -p "$WORK/checkpoints"; [ -e checkpoints ] || ln -s "$WORK/checkpoints" checkpoints
mkdir -p "$WORK/data"
[ -d "$WORK/data/flame_steak_KMOD12T" ] || tar -xf "$BIG/data_slomo_k12_T.tar" -C "$WORK/data"
for d in flame_steak_OPT12T cook_spinach_OPT12T flame_steak_KMOD12T cook_spinach_KMOD12T; do echo "data $d images: $(ls "$WORK/data/$d/images" | wc -l) (want 160)"; done
echo "config md5 (must be 3b8355b189b7dbd064fdec76033bf7c8): $(md5sum configs/default.yaml | cut -d' ' -f1)"
echo "SETUP_REPO_DONE $REPO"
