# CLAUDE.md：給執行這個 repo 的 AI 代理（Claude Code 等）的交付說明

你是這台機器上的執行者。讀完這份檔案後，**不用等任何人確認**，照第 2 節從頭跑到尾，每一步都自己驗證。只有遇到第 4 節列的「必須停下」的狀況才停下並回報。人類使用者可能不在電腦前，請盡量自己解決環境問題。

---

## 1. 目標與完成的定義

**目標**：用官方 4DSloMo 完整流程（stage-1 4DGS → Wan2.1-I2V-14B 影片擴散修補 → stage-2 重訓），以**擴散 50 步**跑完兩個場景 `flame_steak_KMOD12T`、`cook_spinach_KMOD12T`，產生留出相機 cam00 的渲染圖並打包。

**全部做到才算完成：**
1. `OUT/flame_steak_KMOD12T_results.tar`、`OUT/cook_spinach_KMOD12T_results.tar` 和各自的 `.md5`，且 `cd OUT && md5sum -c *.md5` 全部 OK。
2. 兩個場景各有 `OUT/<scene>/cam00_stage2/frame_0100.png … frame_0159.png`：60 張、1352×1014，也就是最終結果；同樣有 `cam00_stage1/` 60 張。
3. 以下檔案都存在：`OUT/preflight.txt`、`OUT/run_all.log`、`work/checkpoints/fetch_weights.log`、`env/pip_freeze_after_build.txt`。
4. 你在 repo 根目錄寫好 `RESULTS_SUMMARY.md`（格式見第 6 節），裡面每一項自我檢查都標明通過或沒通過。
5. 最後告訴使用者：要把哪些檔案（含大小）傳回給 Arthur。

---

## 2. 流程（照順序跑，每步都有自我檢查）

開始前先做第 3 節「判斷進度」：如果之前跑到一半，就從斷點接著跑，**不要重來**。

### 步驟 1：下載官方程式與資料
```bash
bash setup_repo.sh
```
自我檢查：輸出要有 `official code: 898 files match the manifest`（第一次跑才會出現）、4 行 `images: 160 (want 160)`、`config md5 ... 3b8355b189b7dbd064fdec76033bf7c8`（前後兩個值相同），以及 `SETUP_REPO_DONE`。

### 步驟 2：建環境
```bash
bash env/build_env.sh
conda activate 4dslomo      # 非互動 shell 先 source "$(conda info --base)/etc/profile.d/conda.sh"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
自我檢查：build_env.sh 沒有報錯結束，`env/pip_freeze_after_build.txt` 存在，torch 回報 CUDA 可用。

### 步驟 3：下載權重（約 80 GB）
```bash
bash fetch_weights.sh       # 可加 LIMIT=50m 加速；預設 20 MB/s
```
自我檢查：`work/checkpoints/fetch_weights.log` 裡每個檔都有 `DONE` 或 `SKIP (present)`，沒有 `HASH MISMATCH`。下載中斷就重跑，它會續傳。

### 步驟 4：校準（約 1–2 小時）
```bash
OUT_ROOT=$PWD/OUT_calib MAX_CAMS=1 STEPS=5 bash run_all.sh flame_steak_KMOD12T
```
最後出現 `[PART]` 或 `[HOLD]` 是正常的，校準本來就只跑 1 台相機。接著自我檢查：
- a. `OUT_calib/preflight.txt` 沒有 FAIL。
- b. **重現檢查（必須通過）**：`grep S3_RR_PSNR OUT_calib/flame_steak_KMOD12T/logs/stage1_cam00.log` 的值要在 **27.2658 ± 0.3**（也就是 26.97–27.57）之內。
- c. `OUT_calib/flame_steak_KMOD12T/diffusion/chunkA/out/images/` 和 `chunkB/out/images/` 裡，各有 33 張 `cam*_????.png`（第一台相機的 33 幀；正式跑時每段是 20 台 × 33 = 660 張）。打開頭、中、尾各 1 張，和 `OUT_calib/flame_steak_KMOD12T/renders_train/` 裡同一台相機的渲染比較：要看得出是同一個廚房場景（人、牛排、噴槍火焰），不能是雜訊、全黑或單色。
- d. 從 `logs/diffusion_A.log` 讀出擴散每步秒數，執行時用 `nvidia-smi` 看顯存峰值。正式跑的時間估計＝每步秒數 × 50 步 × 40 次（20 台相機 × 2 段）× 2 個場景，再加約 3 小時的訓練與渲染。把這些數字寫進 `RESULTS_SUMMARY.md`。

a、b、c 都通過：執行 `touch CALIBRATED`，**直接進步驟 5**，不用等任何人。
b 沒通過：照第 4 節停下。c 沒通過：照第 4 節處理。

### 步驟 5：正式跑（1 張 48 GB 卡約 32–48 小時）
要在**脫離你這個 session 也會繼續跑**的方式下執行，例如 tmux、nohup 或 sbatch：
```bash
tmux new -d -s slomo "bash -lc 'source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate 4dslomo && cd $PWD && STEPS=50 bash run_all.sh >> run_all_console.log 2>&1'"
```
- 有 2 張卡：在指令前加 `GPU=0,1 NGPU=2`，時間大約減半。
- 機器用 Slurm 排程：改 `slurm_example.sh` 的 `<...>` 後 `sbatch`。時間到被砍就再送一次，會續跑。
- 監看：每 1–3 小時看一次就好，用 `tail -n 20 OUT/run_all.log`，找 `[FAIL]`。不要高頻輪詢。
- 中斷或失敗：修好原因後重跑**同一行**，已完成的階段會自動跳過。

跑完後對**每個場景**自我檢查：
- a. `OUT/<scene>/.done_summary` 存在（各階段的 `.done_*` 都在）。
- b. `cam00_stage2/` 和 `cam00_stage1/` 各 60 張 PNG，尺寸 1352×1014。
- c. stage-1 重現：`logs/stage1_cam00.log` 的 `S3_RR_PSNR` 要在 flame_steak **27.2658 ± 0.3**、cook_spinach **27.5276 ± 0.3** 之內。
- d. `logs/stage2_cam00.log` 有 `S3_RR_PSNR`（沒有門檻，照實記錄），`psnr_stage2.tsv` 有 60 列。
- e. 打開 `cam00_stage2` 的頭、中、尾各 1 張，和 `cam00_stage1` 同一幀比較：要是合理的場景影像，不能是雜訊或全黑。

### 步驟 6：打包與回報
```bash
bash collect_results.sh
cd OUT && md5sum -c *.md5
```
寫好 `RESULTS_SUMMARY.md`（第 6 節），然後告訴使用者要傳回給 Arthur 的檔案清單和大小（第 1 節第 1、3 項，加上 `RESULTS_SUMMARY.md`）。如果頻寬允許，`collect_results.sh` 最後印出的「擴散輸出全部」也一起傳（每個場景約 2 GB）。

---

## 3. 判斷進度（每次開新 session 都先做）
```bash
ls CALIBRATED RESULTS_SUMMARY.md 2>/dev/null
ls OUT_calib/*/.done_* OUT/*/.done_* 2>/dev/null
tail -n 20 OUT/run_all.log 2>/dev/null
tmux ls 2>/dev/null; nvidia-smi
```
各階段依序是：`preflight data stage1_train stage1_cam00 views render_train chunks diffusion_A diffusion_B stage2_data stage2_train stage2_cam00 summary`。如果 `run_all.sh` 還在跑（tmux 或程序還在），**不要再開第二份**，等它跑完。

---

## 4. 決策規則

**可以自己決定（但要記錄在 RESULTS_SUMMARY.md 的「偏離預設」欄）：**
- 環境、編譯問題：調整 conda/pip 套件版本、nvcc 路徑、`TORCH_CUDA_ARCH_LIST` 等，只要不改方法。
- 擴散時顯存不夠（OOM）：可以在 `test_lora.py` 裡啟用 DiffSynth 官方的顯存管理／CPU offload（例如 `pipe.enable_vram_management(num_persistent_param_in_dit=...)`）。數值不變，只是變慢；記錄用了哪個值。
- 有 2 張卡就用 `GPU=0,1 NGPU=2`。
- 機器時間有限：先把 `flame_steak_KMOD12T` 整個跑完，再跑 `cook_spinach_KMOD12T`（`bash run_all.sh flame_steak_KMOD12T`）。
- 權重下載需要登入（401/403）：請使用者 `export HF_TOKEN=...`。不要把 token 印出來，也不要寫進任何檔案。

**必須停下並回報使用者（不要硬跑下去）：**
- 校準或正式跑的 stage-1 `S3_RR_PSNR` 超出 ±0.3。這代表環境和原本不同，跑出來的結果不能拿來比較。請把 `logs/` 的內容、`env/pip_freeze_after_build.txt`、GPU 型號寫進 RESULTS_SUMMARY.md 後停下。
- 擴散輸出是雜訊或全黑，而且你確認權重檔都通過了雜湊檢查。
- 照上面能自己決定的方法試過後，同一個階段還是失敗 3 次。

**絕對不要做：**
- 改 `configs/default.yaml`、解析度、幀數、訓練迭代數、prompt、seed、`ASSIGN`、`ADD_REAL`（都保持預設）。
- 改擴散步數：正式跑一定是 50 步。只有使用者明確要求才可以用 5 步，而且要另外指定 `OUT_ROOT=`，並在摘要裡標明。
- 改官方模型程式、換模型或權重（上面 `test_lora.py` 的顯存設定除外）。
- 刪除或覆蓋 `OUT/`、`OUT_calib/` 裡已完成的內容。`run_all.sh` 會自己把未完成的資料夾改名保留。
- 跑 `*_OPT12T` 兩個臂（不需要）。
- 在共用機器上用 sudo、碰別人的檔案或程序、用沒分配給你的 GPU、佔著 GPU 不用。
- 從非官方來源下載程式或權重；把任何東西推到 git remote（這個 repo 對你是唯讀的）。

---

## 5. 背景（幫你理解在跑什麼，不需要改任何東西）
- 資料：N3DV 多視角影片，共 21 台相機；cam00 是留出的測試相機，不進任何訓練或擴散。另外 20 台訓練相機在測試窗口（第 100–159 幀，共 60 幀）裡，各只有依 K=12 錯開快門排程挑出的 5 張影格（每台約每 12 幀 1 張，子幀時間偏移已寫進時間標籤）。資料夾裡已經是官方 Blender 格式（`transforms_*.json`、`points3d.ply`、`images/`）。
- 流程：stage-1 用 4DGS 訓練 7000 次 → 渲染 20 台訓練相機 × 60 幀 → 切成 A 段（100–132）和 B 段（127–159），裁成 1024²，每台相機每段用 Wan2.1-I2V-14B + 4DSloMo LoRA 修補一次 → 用修補結果組成 stage-2 資料集重訓 → 渲染 cam00 的 60 幀。
- 你只負責產生渲染圖。和真值的比較由 Arthur 那邊做。
- 細節：`README.md`（給人看的步驟）、`EXPECTED_OUTPUTS.md`（每個輸出檔的說明）、`code/CHANGES.md`（兩個 patch 的理由）、`DECISIONS_NEEDED.md`（設定選擇的理由）。

---

## 6. RESULTS_SUMMARY.md 格式
```markdown
# 4DSloMo full × N3DV K=12 結果摘要
- 狀態：完成 / 未完成（停在哪一步、原因）
- 機器：GPU 型號×張數、顯存、驅動、CUDA；RAM；作業系統
- repo commit：<git rev-parse HEAD>
- 擴散步數：50（或標明）
| 場景 | stage-1 S3_RR_PSNR（基準） | stage-2 S3_RR_PSNR | 擴散 s/步 | 顯存峰值 | 各階段時間（stage1_train / diffusion_A / diffusion_B / stage2_train） |
|---|---|---|---|---|---|
| flame_steak_KMOD12T | x（27.2658） | y | | | |
| cook_spinach_KMOD12T | x（27.5276） | y | | | |
- 校準：S3_RR_PSNR、s/步、顯存峰值、估計總時數
- 自我檢查：步驟 1–6 每一項 通過/沒通過
- 偏離預設：（沒有就寫「無」）
- 要傳回的檔案：路徑＋大小＋md5
```
