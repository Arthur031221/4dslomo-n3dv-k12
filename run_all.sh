#!/usr/bin/env bash
# run_all.sh [SCENE_ARM ...]      SCENE_ARM in {flame_steak_OPT12T cook_spinach_OPT12T flame_steak_KMOD12T cook_spinach_KMOD12T}
# default (primary, v2 11-06): flame_steak_KMOD12T cook_spinach_KMOD12T (= the arm of all old stage-1 runs; reproduction targets in EXPECTED_OUTPUTS.md).
#   OPT12 (secondary, optional, low priority): pass flame_steak_OPT12T cook_spinach_OPT12T.
# A full run (MAX_CAMS=0) refuses to start until the calibration run has been done and $BUNDLE/CALIBRATED exists (CLAUDE.md step 4 / README 2.5).
# Full official 4DSloMo pipeline on our N3DV K=12 staggered protocol (README "Quick Start" 1-3), resumable: every stage writes OUT/<sa>/.done_<stage>
# only on success and is skipped on re-run; an unfinished output dir is MOVED aside (*.incomplete_<time>), finished outputs are never overwritten.
#
# env (all optional):  STEPS=50 (diffusion steps; official default 50, README says 5 gives ~80% of the quality)  GPU=0 (or "0,1" with NGPU=2)  NGPU=1
#   ASSIGN=drop0|keep0|both  ADD_REAL=0|1   MAX_CAMS=N (diffusion on the first N cameras only = calibration; stage is then NOT marked done)
#   STOP_AFTER=<stage name> (exit after that stage, e.g. STOP_AFTER=diffusion_A)   PROMPT="..." (overrides the per-scene prompt)
#   WORK DATA_ROOT OUT_ROOT REPO (paths)   SEED=0 (diffusion noise seed; official 0)
# stages: preflight data stage1_train stage1_cam00 views render_train chunks diffusion_A diffusion_B stage2_data stage2_train stage2_cam00 summary
set -uo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-$BUNDLE/work}; REPO=${REPO:-$WORK/4DSloMo}; DATA_ROOT=${DATA_ROOT:-$WORK/data}; OUT_ROOT=${OUT_ROOT:-$BUNDLE/OUT}
STEPS=${STEPS:-50}; GPU=${GPU:-0}; NGPU=${NGPU:-1}; ASSIGN=${ASSIGN:-drop0}; ADD_REAL=${ADD_REAL:-0}; SEED=${SEED:-0}; MAX_CAMS=${MAX_CAMS:-0}
PY=${PY:-python}; TOOLS=$BUNDLE/tools; CFG_MD5=3b8355b189b7dbd064fdec76033bf7c8
export CUDA_VISIBLE_DEVICES=$GPU TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9} TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$WORK/torch_ext} \
       OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT" "$TORCH_EXTENSIONS_DIR"
SAS=("$@"); [ ${#SAS[@]} -gt 0 ] || SAS=(flame_steak_KMOD12T cook_spinach_KMOD12T)
MASTER=$OUT_ROOT/run_all.log
say() { echo "$(date +%F_%T) $*" | tee -a "$MASTER"; }

must() { "$@" || { echo "FAILED rc=$?: $*" >&2; exit 97; }; }
cdr() { cd "$REPO" || exit 96; }
fresh_dir() {   # fresh_dir PATH : an existing, unfinished dir is moved aside (never deleted)
  if [ -e "$1" ]; then mv "$1" "$1.incomplete_$(date +%m%d_%H%M%S)"; fi
}
stage() {       # stage NAME cmd... : skip when .done_NAME exists; log to OUT/<sa>/logs/NAME.log
  local name=$1; shift
  if [ -e "$SA_OUT/.done_$name" ]; then say "[skip] $SA $name"; return 0; fi
  say "[run ] $SA $name"; local t0=$(date +%s)
  ( "$@" ) >> "$SA_OUT/logs/$name.log" 2>&1; local rc=$?
  if [ $rc -ne 0 ]; then say "[FAIL] $SA $name rc=$rc  (tail -n 40 $SA_OUT/logs/$name.log)"; return $rc; fi
  if [ "$name" = diffusion_A ] || [ "$name" = diffusion_B ]; then
    local ch=${name#diffusion_}; local n; n=$(ls "$SA_OUT/diffusion/chunk$ch/out/images/"cam*_????.png 2>/dev/null | wc -l)
    if [ "$n" -ne 660 ]; then say "[PART] $SA $name: $n/660 frames present (MAX_CAMS or a skipped camera) -> not marked done; wall_s=$(( $(date +%s) - t0 ))"; return 0; fi
  fi
  echo "$(date +%F_%T) wall_s=$(( $(date +%s) - t0 ))" > "$SA_OUT/.done_$name"; say "[done] $SA $name wall_s=$(( $(date +%s) - t0 ))"
  if [ "${STOP_AFTER:-}" = "$name" ]; then say "STOP_AFTER=$name reached"; exit 0; fi
  return 0
}
scratch_model() {   # scratch_model SRC_MODEL DST : cfg_args copy so that renders never write into a finished model dir (Scene() rewrites input.ply / cameras.json)
  mkdir -p "$2"; cp "$1/cfg_args" "$2/cfg_args"
}
prompt_of() {
  if [ -n "${PROMPT:-}" ]; then echo "$PROMPT"; return; fi
  case $1 in flame_steak*) echo "A person cooking a steak in a kitchen";; cook_spinach*) echo "A person cooking spinach in a kitchen";; *) echo "A person cooking in a kitchen";; esac
}

# ----------------------------------------------------------------- preflight (once)
pre() {
  { echo "== $(date)"; echo "host $(hostname)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; $PY - <<'PYC'
import sys, torch
print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
for m in ('cv2', 'diffsynth', 'plyfile', 'omegaconf', 'imagesize', 'simple_knn', 'pointops2', 'numpy', 'PIL'):
    try: __import__(m); print('OK  ', m)
    except Exception as e: print('FAIL', m, e)
PYC
    echo "default.yaml md5 $(md5sum "$REPO/configs/default.yaml" | cut -d' ' -f1)"
    for f in models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth 4DSloMo_LoRA.ckpt diffusion_pytorch_model-00007-of-00007.safetensors google/umt5-xxl/spiece.model; do
      [ -s "$REPO/checkpoints/$f" ] && echo "weights OK   $f" || echo "weights MISSING $f"; done
    ls "$REPO/render_views.py" "$REPO/render_s3_v2.py"; } > "$OUT_ROOT/preflight.txt" 2>&1
  cat "$OUT_ROOT/preflight.txt"
  [ "$(md5sum "$REPO/configs/default.yaml" | cut -d' ' -f1)" = "$CFG_MD5" ] || { echo "default.yaml md5 differs from the one used for all old stage-1 runs"; return 3; }
  grep -q "MISSING\|FAIL" "$OUT_ROOT/preflight.txt" && { echo "preflight problems (see above)"; return 4; }
  return 0
}
say "run_all start: ${SAS[*]} STEPS=$STEPS GPU=$GPU ASSIGN=$ASSIGN ADD_REAL=$ADD_REAL"
pre || { say "PREFLIGHT FAILED"; exit 4; }
if [ "$MAX_CAMS" = 0 ] && [ ! -e "$BUNDLE/CALIBRATED" ]; then
  say "REFUSED: full run before calibration. First: OUT_ROOT=$BUNDLE/OUT_calib MAX_CAMS=1 STEPS=5 bash run_all.sh flame_steak_KMOD12T ; self-check it (CLAUDE.md step 4: S3_RR_PSNR within 27.2658 +-0.3, 33 images per chunk), then: touch $BUNDLE/CALIBRATED"; exit 6
fi

RC_ALL=0
for SA in "${SAS[@]}"; do
  case $SA in flame_steak_OPT12T|cook_spinach_OPT12T|flame_steak_KMOD12T|cook_spinach_KMOD12T) ;; *) say "bad SCENE_ARM $SA"; RC_ALL=2; continue;; esac
  SA_OUT=$OUT_ROOT/$SA; DATA=$DATA_ROOT/$SA; mkdir -p "$SA_OUT/logs"; M1=$SA_OUT/stage1; M2=$SA_OUT/stage2
  PROMPT_USE=$(prompt_of "$SA"); echo "$PROMPT_USE" > "$SA_OUT/prompt.txt"
  (   # NOTE: no `set -e` here (it is ignored inside `stage ... || exit` anyway); every command inside the stage functions goes through `must`
    # ---- data: check the unpacked dataset (setup_repo.sh untars data_slomo_k12_T.tar into $DATA_ROOT)
    chk_data() { [ -f "$DATA/transforms_train.json" ] && [ "$(ls "$DATA/images" | wc -l)" = 160 ] && [ -f "$DATA/points3d.ply" ] && echo "data OK $DATA"; }
    stage data chk_data || exit 1

    # ---- stage 1: official train.py, configs/default.yaml unchanged, 7000 it (README step 1)
    train1() { fresh_dir "$M1"; must mkdir -p "$M1"; must cp "$REPO/configs/default.yaml" "$M1/default.yaml.used"
               cdr; must $PY -B train.py --config configs/default.yaml --model_path "$M1" --source_path "$DATA"
               must test -e "$M1/chkpnt7000.pth"; }
    stage stage1_train train1 || exit 1
    # ---- held-out cam00 of the stage-1 model (white bg = training bg), same renderer as the old stage-1 runs; prints S3_RR_PSNR
    cam00_s1() { fresh_dir "$SA_OUT/cam00_stage1"; scratch_model "$M1" "$SA_OUT/scratch_s1"; cdr
                 must $PY -B render_s3_v2.py --model_path "$SA_OUT/scratch_s1" --source_path "$DATA" --loaded_pth "$M1/chkpnt7000.pth" --out_dir "$SA_OUT/cam00_stage1" --config "$M1/default.yaml.used" --check_psnr
                 [ "$(ls "$SA_OUT/cam00_stage1"/frame_????.png | wc -l)" = 60 ]; }
    stage stage1_cam00 cam00_s1 || exit 1

    # ---- render the TRAINING cameras at all 60 frames (stands in for README `render.py`; green bg as official render.py l.68)
    mkviews() { fresh_dir "$SA_OUT/views_src"; must $PY -I "$TOOLS/mk_views.py" "$DATA" "$SA_OUT/views_src"; }
    stage views mkviews || exit 1
    rend() { mkdir -p "$SA_OUT/renders_train"; scratch_model "$M1" "$SA_OUT/scratch_views"; cdr
             must $PY -B render_views.py --model_path "$SA_OUT/scratch_views" --source_path "$SA_OUT/views_src" --loaded_pth "$M1/chkpnt7000.pth" --out_dir "$SA_OUT/renders_train" --config "$M1/default.yaml.used" --render_bg green --skip_existing
             [ "$(ls "$SA_OUT/renders_train"/cam??_????.png | wc -l)" = 1200 ]; }
    stage render_train rend || exit 1

    # ---- chunks: A = frames 100..132, B = frames 127..159; official process_video.py (README step 2b) on each
    mkchunks() { for ch in A B; do
                   fresh_dir "$SA_OUT/diffusion/chunk$ch"; must mkdir -p "$SA_OUT/diffusion/chunk$ch"
                   must $PY -I "$TOOLS/prep_chunks.py" "$DATA" "$SA_OUT/renders_train" "$SA_OUT/diffusion/chunk$ch" $ch
                   cdr; must $PY process_video.py --input_folder "$SA_OUT/diffusion/chunk$ch/in/test/ours_None" --max_frames 33
                   [ "$(ls "$SA_OUT/diffusion/chunk$ch/in/test/ours_None/video_crop"/cam??.mp4 | wc -l)" = 20 ] || exit 95
                   [ "$(ls "$SA_OUT/diffusion/chunk$ch/in/test/ours_None/gt_crop"/cam??_0000.png | wc -l)" = 20 ] || exit 95
                 done; }
    stage chunks mkchunks || exit 1

    # ---- diffusion (README step 2c): 20 cameras x 2 chunks; resumable per camera (--skip_existing)
    diff_run() { local ch=$1 cams="@$SA_OUT/diffusion/chunk$1/cams.txt"
                 if [ "$MAX_CAMS" -gt 0 ]; then head -n "$MAX_CAMS" "$SA_OUT/diffusion/chunk$ch/cams.txt" > "$SA_OUT/diffusion/chunk$ch/cams_run.txt"; cams="@$SA_OUT/diffusion/chunk$ch/cams_run.txt"; fi
                 cdr; must $PY -m torch.distributed.run --standalone --nproc_per_node="$NGPU" test_lora.py \
                   --input_folder "$SA_OUT/diffusion/chunk$ch/in" --output_folder "$SA_OUT/diffusion/chunk$ch/out" --model_path checkpoints/4DSloMo_LoRA.ckpt \
                   --num_inference_steps "$STEPS" --prompt "$PROMPT_USE" --cams "$cams" --num_frames 33 --seed "$SEED" --skip_existing; }
    stage diffusion_A diff_run A || exit 1
    stage diffusion_B diff_run B || exit 1
    if [ ! -e "$SA_OUT/.done_diffusion_A" ] || [ ! -e "$SA_OUT/.done_diffusion_B" ]; then say "[HOLD] $SA diffusion incomplete (MAX_CAMS?) -> stopping this scene/arm"; exit 0; fi

    # ---- stage-2 dataset (README step 3 "prepare camera pose and timestamp", generated instead of shipped), retrain, render cam00
    mk2() { fresh_dir "$SA_OUT/stage2_data"; local ex=(); [ "$ADD_REAL" = 1 ] && ex=(--add_real)
            must $PY -I "$TOOLS/assemble_stage2.py" "$DATA" "$SA_OUT/diffusion" "$SA_OUT/stage2_data" --assign "$ASSIGN" "${ex[@]}"; }
    stage stage2_data mk2 || exit 1
    train2() { fresh_dir "$M2"; must mkdir -p "$M2"; must cp "$REPO/configs/default.yaml" "$M2/default.yaml.used"
               cdr; must $PY -B train.py --config configs/default.yaml --model_path "$M2" --source_path "$SA_OUT/stage2_data"
               must test -e "$M2/chkpnt7000.pth"; }
    stage stage2_train train2 || exit 1
    cam00_s2() { fresh_dir "$SA_OUT/cam00_stage2"; scratch_model "$M2" "$SA_OUT/scratch_s2"; cdr
                 must $PY -B render_s3_v2.py --model_path "$SA_OUT/scratch_s2" --source_path "$SA_OUT/stage2_data" --loaded_pth "$M2/chkpnt7000.pth" --out_dir "$SA_OUT/cam00_stage2" --config "$M2/default.yaml.used" --check_psnr
                 [ "$(ls "$SA_OUT/cam00_stage2"/frame_????.png | wc -l)" = 60 ]; }
    stage stage2_cam00 cam00_s2 || exit 1
    summ() { for s in 1 2; do must $PY -I "$TOOLS/eval_cam00.py" "$SA_OUT/cam00_stage$s" "$DATA" "$SA_OUT/psnr_stage$s.tsv"; done; }
    stage summary summ || true
  ) || { say "[ABORT] $SA (re-run run_all.sh to resume)"; RC_ALL=1; }
done
say "run_all end rc=$RC_ALL"; exit $RC_ALL
