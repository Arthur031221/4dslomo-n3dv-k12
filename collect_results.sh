#!/usr/bin/env bash
# collect_results.sh [SCENE_ARM ...]  -> OUT/<sa>_results.tar + .md5   (what to send back; the big 1200-image renders_train and the full diffusion dirs are NOT included)
set -euo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); OUT_ROOT=${OUT_ROOT:-$BUNDLE/OUT}
SAS=("$@"); [ ${#SAS[@]} -gt 0 ] || SAS=(flame_steak_KMOD12T cook_spinach_KMOD12T)
for sa in "${SAS[@]}"; do
  cd "$OUT_ROOT"
  [ -d "$sa" ] || { echo "no $sa"; continue; }
  files=("$sa"/cam00_stage1 "$sa"/cam00_stage2 "$sa"/logs "$sa"/prompt.txt "$sa"/psnr_stage1.tsv "$sa"/psnr_stage2.tsv "$sa"/.done_* \
         "$sa"/stage1/chkpnt7000.pth "$sa"/stage1/cfg_args "$sa"/stage1/default.yaml.used "$sa"/stage2/chkpnt7000.pth "$sa"/stage2/cfg_args "$sa"/stage2/default.yaml.used \
         "$sa"/diffusion/chunkA/plan.json "$sa"/diffusion/chunkB/plan.json "$sa"/stage2_data/transforms_train.json "$sa"/stage2_data/transforms_test.json "$sa"/views_src/transforms_test.json "$OUT_ROOT/preflight.txt" "$OUT_ROOT/run_all.log")
  ex=(); for f in "${files[@]}"; do [ -e "$f" ] && ex+=("$f"); done
  # a few diffusion outputs for visual inspection (cam01, cam10, cam20; both chunks)
  for ch in A B; do for c in cam01 cam10 cam20; do for f in "$sa"/diffusion/chunk$ch/out/images/${c}_00*.png; do [ -e "$f" ] && ex+=("$f"); done; done; done
  tar -cf "$OUT_ROOT/${sa}_results.tar" "${ex[@]}"; (cd "$OUT_ROOT" && md5sum "${sa}_results.tar" > "${sa}_results.tar.md5"; cat "${sa}_results.tar.md5"; ls -l "${sa}_results.tar")
  echo "OPTIONAL (about 2 GB, bandwidth permitting): tar -cf ${sa}_diffusion_full.tar $sa/diffusion/chunkA/out $sa/diffusion/chunkB/out"
done
