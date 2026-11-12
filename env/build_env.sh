#!/usr/bin/env bash
# Build the 4DSloMo environment exactly as the OFFICIAL README (l.36-44) says, split into explicit steps.
# usage: bash env/build_env.sh [REPO_DIR]     (REPO_DIR = unpacked + patched repo made by setup_repo.sh; default <bundle>/work/4DSloMo)
# Official README:   conda create -n 4dslomo python=3.10 -y ; conda activate 4dslomo
#                    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
#                    pip install -r requirements.txt
# env/environment.yml is the OLD fudan-zvg 4DGS file (python 3.7 / torch 1.12): shipped verbatim for completeness, NOT used (README does not use it).
set -euo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=${1:-$BUNDLE/work/4DSloMo}
[ -f "$REPO/requirements.txt" ] || { echo "run setup_repo.sh first ($REPO)"; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n 4dslomo python=3.10 -y
conda activate 4dslomo
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# --- nvcc: the CUDA extensions (simple-knn, pointops2) and the rasterizer (JIT) must be compiled with an nvcc whose MAJOR version equals torch's (11.x) ---
if ! command -v nvcc >/dev/null || ! nvcc --version | grep -q "release 11\."; then
  echo "no nvcc 11.x on PATH -> installing the CUDA 11.8 toolkit into the conda env (about 4-5 GB). Skip this by exporting CUDA_HOME to a system CUDA 11.x."
  conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
  export CUDA_HOME="$CONDA_PREFIX"
else
  export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"      # RTX 6000 Ada = sm_89 (default); H200 = 9.0: export it before running (README_c.md)
cd "$REPO"
# requirements.txt lines 1-9 are plain pip packages; the last two lines are the local CUDA extensions.
pip install $(grep -v '^\./' requirements.txt | tr '\n' ' ')
pip install -r "$BUNDLE/env/constraints_extra.txt"
# local extensions of the official requirements.txt ("./simple-knn", "./pointops2"); --no-build-isolation because their setup.py imports torch
pip install --no-build-isolation ./simple-knn ./pointops2
# diff-gaussian-rasterization is NOT pip-installed: its setup.py lists a package dir that does not exist. The official
# gaussian_renderer/diff_gaussian_rasterization.py l.15-25 JIT-compiles it with torch.utils.cpp_extension.load at first import (needs nvcc + ninja + CUDA_HOME).
# Pre-build it now (on the GPU machine, fixed cache dir, takes a few minutes):
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BUNDLE/work/torch_ext}"; mkdir -p "$TORCH_EXTENSIONS_DIR"
python -B - <<'PY'
import importlib.util, os, torch
sp = importlib.util.spec_from_file_location('slomo_rast', os.path.join(os.getcwd(), 'gaussian_renderer', 'diff_gaussian_rasterization.py'))
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
print('RAST_JIT_OK', m._C)
PY
python - <<'PY'
import torch, simple_knn, importlib
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))
for m in ('cv2', 'diffsynth', 'plyfile', 'omegaconf', 'imagesize', 'kornia', 'transformers', 'torchmetrics', 'pointops2'):
    try: mod = importlib.import_module(m); print('OK ', m, getattr(mod, '__version__', ''))
    except Exception as e: print('FAIL', m, e)
PY
echo "ENV_BUILD_DONE (pip freeze -> $BUNDLE/env/pip_freeze_after_build.txt)"; pip freeze > "$BUNDLE/env/pip_freeze_after_build.txt"
