#!/usr/bin/env bash
# fetch_weights.sh [DEST]   -- downloads the model weights ONE FILE AT A TIME, rate limited, HTTPS only, skipping files already present.
#   Wan-AI/Wan2.1-I2V-14B-720P  (only the files test_lora.py l.46-65 + FixModel.py l.131 load)   https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P
#   yutian05/4DSloMo            (4DSloMo_LoRA.ckpt, test_lora.py l.27 / README l.45-49)           https://huggingface.co/yutian05/4DSloMo
# env: LIMIT (curl --limit-rate, default 20m = 20 MB/s) ; HF_TOKEN (optional, read from the environment only, never printed or logged)
# Log: DEST/fetch_weights.log  (one line per file: time, name, bytes, sha256, md5). Re-running only fetches what is missing; partial files resume (.part).
# Expected total size is about 80+ GB (the 7 DiT shards are the bulk) -- NOT verified, see README.
set -euo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${1:-$BUNDLE/work/checkpoints}; LIMIT=${LIMIT:-20m}
mkdir -p "$DEST/google/umt5-xxl"; LOG="$DEST/fetch_weights.log"; touch "$LOG"
log() { echo "$(date +%F_%T) $*" | tee -a "$LOG"; }
auth_cfg() { if [ -n "${HF_TOKEN:-}" ]; then printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"; else printf '# no token\n'; fi; }
# x-linked-etag: 64 hex = sha256 of an LFS file; 40 hex = git blob SHA-1 of a small non-LFS file (sha1 of "blob <size>\\0<content>"); empty = nothing to check.
check_hash() {   # check_hash PATH EXPECTED SHA256
  case ${#2} in
    0)  return 0 ;;
    64) [ "$2" = "$3" ] ;;
    40) [ "$2" = "$( { printf 'blob %s\0' "$(stat -c %s "$1")"; cat "$1"; } | sha1sum | cut -d' ' -f1)" ] ;;
    *)  return 1 ;;
  esac
}
fetch() {   # fetch REPO FILE  -> DEST/FILE
  local repo=$1 file=$2 url out
  url="https://huggingface.co/${repo}/resolve/main/${file}"; out="$DEST/$file"
  if [ -s "$out" ] && [ ! -e "$out.part" ]; then log "SKIP (present) $file"; return 0; fi
  log "START $repo/$file"
  # expected sha256 of LFS files = x-linked-etag of the first (redirect) response; small git files have none
  local exp; exp=$(auth_cfg | curl -sI --proto '=https' --tlsv1.2 -K - "$url" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-linked-etag"{gsub(/"/,"",$2); print $2}' | head -n1 || true)
  if [ -s "$out.part" ] && [ -n "$exp" ] && check_hash "$out.part" "$exp" "$(sha256sum "$out.part" | cut -d' ' -f1)"; then
    log "part already complete $file"
  else
    auth_cfg | curl -fL --proto '=https' --tlsv1.2 --limit-rate "$LIMIT" --retry 5 --retry-delay 10 -C - -K - -o "$out.part" "$url"
  fi
  local sha; sha=$(sha256sum "$out.part" | cut -d' ' -f1)
  if ! check_hash "$out.part" "$exp" "$sha"; then log "HASH MISMATCH $file expected=$exp sha256=$sha (kept as $out.part)"; return 1; fi
  mv "$out.part" "$out"
  log "DONE $file bytes=$(stat -c %s "$out") sha256=$sha md5=$(md5sum "$out" | cut -d' ' -f1) expected_sha256=${exp:-none}"
}
WAN=Wan-AI/Wan2.1-I2V-14B-720P
for f in models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth \
         diffusion_pytorch_model-0000{1,2,3,4,5,6,7}-of-00007.safetensors \
         google/umt5-xxl/special_tokens_map.json google/umt5-xxl/spiece.model google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json; do
  fetch $WAN "$f"
done
if ! fetch yutian05/4DSloMo 4DSloMo_LoRA.ckpt; then
  log "4DSloMo_LoRA.ckpt not found at yutian05/4DSloMo. Listing the repo files ONCE (single API call) -- put the right file in $DEST/4DSloMo_LoRA.ckpt:"
  python - <<'PY' 2>&1 | tee -a "$LOG"
from huggingface_hub import list_repo_files
print(list_repo_files('yutian05/4DSloMo'))
PY
  exit 1
fi
log "FETCH_WEIGHTS_DONE"
