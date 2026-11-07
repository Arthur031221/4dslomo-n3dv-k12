#!/bin/bash
# 只有在機器用 Slurm 排程時才需要（直接登入就能用 GPU 的機器請照 README 用 tmux）。
# <...> 要換成你帳號/機器的值；改完用 `sbatch slurm_example.sh` 送出，`squeue -u $USER` 看狀態。
#SBATCH --job-name=slomo_k12
#SBATCH --partition=<GPU 分區名稱>
#SBATCH --account=<計畫代號>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate 4dslomo
# 校準用：OUT_ROOT=$PWD/OUT_calib MAX_CAMS=1 STEPS=5 bash run_all.sh flame_steak_KMOD12T
STEPS=50 bash run_all.sh      # 時間到被中斷也沒關係：重新 sbatch 會從斷點續跑
