# 4DSloMo 完整版 × N3DV（K=12 錯開快門）執行包

> **兩台機器（11-12 追加說明 c）：** 48 GB 舊機器和 H200 新機器分工平行跑，看 `README_c.md`。

> **追加包（11-12）：再多跑 6 個 N3DV 場景，流程和先前完全相同。** 說明看 `README_b.md`；新資料 `data_slomo_k12_T_b.tar`（約 1.22 GB）在本 repo 的 Release **v2**（tag `v2-data-b`），md5 見 `MD5SUMS_big_b.txt`。環境、權重、校準都沿用舊的，不用重做。照 `README_b.md` 的順序跑，**每跑完一個就先回傳那一個**；台北時間 11-15 24:00 之後才完成的就不必再跑。


這個 repo 是一份「照著跑就好」的執行包：用**官方 4DSloMo**（SIGGRAPH Asia 2025；stage-1 4DGS → Wan2.1-I2V-14B 影片擴散修補 → stage-2 重訓）在 N3DV 的兩個場景（`flame_steak`、`cook_spinach`）上跑完整流程，最後輸出留出相機 cam00 的 60 張渲染圖。

你要做的事：照第 2 節一步一步跑：先校準、自己核對、通過就直接跑正式版，最後把結果打包傳回給我。

**用 Claude Code（或其他 AI 代理）跑最省事：** clone 下來後在 repo 資料夾裡開 `claude`，跟它說「照 CLAUDE.md 把整個流程跑完」。`CLAUDE.md` 寫好了目標、每一步的自我檢查、什麼情況要停下來；不用等我確認。

---

## 0. 先看這三點

1. **一定要先校準（約 1–2 小時）。** 照 2.5 的三項自己核對，全部通過就 `touch CALIBRATED`、直接跑正式版，不用等我。沒有這個檔案，正式跑會自己拒絕開始（這是故意的，避免白跑 40 小時）。
2. **不要改** `configs/default.yaml`、解析度、幀數、模型程式。要調的只有第 2 節列出的環境變數。
3. **中斷了不用怕。** 重新執行同一行指令就會從斷點續跑；已完成的結果不會被刪或覆蓋（未完成的資料夾會被改名成 `*.incomplete_<時間>`）。不要手動刪 `OUT*/` 裡的東西。

---

## 1. 需要什麼

| 項目 | 需求 |
|---|---|
| GPU | 1 張 48 GB（RTX 6000 Ada 為目標機種）；2 張也可以（見 2.6） |
| 系統 | Linux x86_64，NVIDIA 驅動可跑 CUDA 11.8 |
| 軟體 | conda（Miniconda 即可）、git、curl、patch、tmux（建議） |
| 磁碟 | **≥ 200 GB**（權重約 80 GB、每個場景輸出約 15–20 GB） |
| 記憶體 | ≥ 64 GB RAM |
| 網路 | 只有安裝時需要：GitHub、HuggingFace、conda/pip。跑實驗本身不連網 |
| HuggingFace 帳號 | 通常不需要（權重是公開的）。若下載出現 401/403，才需要設 `HF_TOKEN`（見 2.4） |

---

## 2. 步驟

### 2.1 下載本 repo

```bash
git clone https://github.com/Arthur031221/4dslomo-n3dv-k12.git
cd 4dslomo-n3dv-k12
md5sum -c MD5SUMS          # 每一行都要是 OK
```

### 2.2 下載官方程式與資料、套 patch（約 5–10 分鐘）

```bash
bash setup_repo.sh
```

它會自動：
- 從官方 GitHub 下載 4DSloMo 原始碼（固定 commit `9f75dd56f34a3440d3333bc048b9e2d5b4497cf3`，約 290 MB），逐檔核對 md5；
- 從本 repo 的 Release 下載資料 `data_slomo_k12_T.tar`（約 630 MB），核對 md5；
- 套上 `code/patches/` 的兩個小 patch（說明見 `code/CHANGES.md`），把工具放進官方程式資料夾。

**成功的樣子**（最後幾行）：
```
data flame_steak_OPT12T images: 160 (want 160)
data cook_spinach_OPT12T images: 160 (want 160)
data flame_steak_KMOD12T images: 160 (want 160)
data cook_spinach_KMOD12T images: 160 (want 160)
config md5 (must be 3b8355b189b7dbd064fdec76033bf7c8): 3b8355b189b7dbd064fdec76033bf7c8
SETUP_REPO_DONE .../work/4DSloMo-9f75dd56f34a3440d3333bc048b9e2d5b4497cf3
```
如果出現 `MISMATCH`，照訊息刪掉該檔重跑；還是不行就停下來告訴我。

### 2.3 建 conda 環境（約 20–40 分鐘）

```bash
bash env/build_env.sh       # 建立 conda 環境 "4dslomo"（python 3.10、torch 2.4.1+cu118、編譯 simple-knn / pointops2 / rasterizer）
conda activate 4dslomo
```
跑完會寫 `env/pip_freeze_after_build.txt`（最後要一起傳回）。

### 2.4 下載模型權重（約 80 GB；20 MB/s 約 1–1.5 小時）

```bash
bash fetch_weights.sh       # 一次下載一個檔、會限速、已存在的跳過、可續傳；日誌 work/checkpoints/fetch_weights.log
```
- 想快一點：`LIMIT=50m bash fetch_weights.sh`（預設 20m = 20 MB/s）。
- 出現 401/403：到 HuggingFace 申請 token 後 `export HF_TOKEN=<你的 token>` 再跑一次（腳本只從環境變數讀，不會印出或寫進日誌）。
- `4DSloMo_LoRA.ckpt` 出現 404：腳本會列出該 repo 的檔案清單，把正確的檔案放到 `work/checkpoints/4DSloMo_LoRA.ckpt`。

### 2.5 校準（必做，約 1–2 小時）

只跑 1 個場景、每段 1 台相機、擴散 5 步，寫到另一個資料夾 `OUT_calib`：

```bash
tmux new -s calib            # 建議在 tmux 裡跑，斷線不會停
conda activate 4dslomo
OUT_ROOT=$PWD/OUT_calib MAX_CAMS=1 STEPS=5 bash run_all.sh flame_steak_KMOD12T
```
跑到最後會印 `[PART]`／`[HOLD]` 然後停，這是正常的（校準只跑 1 台相機）。接著**自己核對三項**：

```bash
grep S3_RR_PSNR OUT_calib/flame_steak_KMOD12T/logs/stage1_cam00.log        # (1) 要在 27.2658 ± 0.3 之內（26.97–27.57）
ls OUT_calib/flame_steak_KMOD12T/diffusion/chunk{A,B}/out/images/ | head   # (2) A、B 各 33 張 cam*_????.png
tail -n 3 OUT_calib/flame_steak_KMOD12T/logs/diffusion_A.log               # 擴散每步秒數（記下來，用來估正式跑時間）
```
(3) 打開幾張擴散輸出，和 `OUT_calib/flame_steak_KMOD12T/renders_train/` 同一台相機的圖比：要看得出是同一個廚房場景，不能是雜訊或全黑。

- 三項都通過 → 進 2.6。
- (1) 沒過 → 停下，把 `logs/`、`env/pip_freeze_after_build.txt`、GPU 型號傳給我（環境和原本不同，結果不能比）。
- (2)(3) 沒過 → 先看第 4 節；還是不行再傳日誌給我。

### 2.6 正式跑（校準三項通過後直接跑）

```bash
touch CALIBRATED
tmux new -s slomo
conda activate 4dslomo
STEPS=50 bash run_all.sh 2>&1 | tee -a run_all_console.log      # 預設跑 flame_steak_KMOD12T 與 cook_spinach_KMOD12T
```
- 有 2 張卡：`GPU=0,1 NGPU=2 STEPS=50 bash run_all.sh`。
- 斷線：`tmux attach -t slomo` 回去看；中斷了就重跑同一行。
- 機器是 Slurm 排程系統（要用 `sbatch` 的那種）：改 `slurm_example.sh` 裡 `<...>` 的值後 `sbatch slurm_example.sh`。時間到被砍也沒關係，再送一次會續跑。
- 正式跑一定是 50 步。只有在我跟你說要跑 5 步版時才用 `STEPS=5`（而且要另指 `OUT_ROOT=`，不同步數不能混在同一個資料夾）。
- 其他旗標（`ASSIGN`、`ADD_REAL`、`PROMPT`、`STOP_AFTER`）**不用動**，保持預設。

### 2.7 打包傳回

```bash
bash collect_results.sh
```
**請傳回：**
- `OUT/flame_steak_KMOD12T_results.tar`、`OUT/cook_spinach_KMOD12T_results.tar` 和各自的 `.md5`（內含 cam00 stage-1 / stage-2 各 60 張圖、模型 checkpoint、日誌）；
- `OUT/preflight.txt`、`OUT/run_all.log`、`work/checkpoints/fetch_weights.log`、`env/pip_freeze_after_build.txt`。

傳之前先自己核一次：`cd OUT && md5sum -c *.md5`，並確認兩個場景的 `cam00_stage2/` 各有 60 張圖、`logs/stage1_cam00.log` 的 `S3_RR_PSNR` 在 flame_steak 27.2658 ± 0.3、cook_spinach 27.5276 ± 0.3 之內。若頻寬允許，擴散輸出全部（每個場景約 2 GB）也請一起傳，指令會印在 `collect_results.sh` 的最後。

---

## 3. 時間與空間估計（RTX 6000 Ada 48 GB × 1）

| 階段 | 每個場景 | 備註 |
|---|---|---|
| stage-1 訓練（7000 it） | 0.5–0.8 h | |
| 渲染（60 + 1200 張） | 0.1–0.2 h | |
| **擴散修補，50 步** | **約 15–22 h** | 每步約 28–30 秒（推算：同一程式在 RTX 4090 上實測 45 s/步，但 4090 必須把模型部分放到 CPU；48 GB 卡整個模型放得下） |
| stage-2 訓練（7000 it） | 0.4–0.8 h | |
| **一個場景合計** | **約 16–24 h** | |
| **兩個場景（正式跑）** | **約 32–48 h** | 2 張卡大約減半 |
| 校準（5 步、1 台相機） | 約 1–2 h | |

磁碟：權重約 80 GB、資料 0.6 GB、每個場景輸出 15–20 GB → 建議至少 200 GB。

---

## 4. 出問題怎麼辦

| 狀況 | 怎麼做 |
|---|---|
| `OUT/preflight.txt` 說缺權重／缺套件／config md5 不符 | 照訊息補；config md5 必須是 `3b8355b189b7dbd064fdec76033bf7c8` |
| 擴散階段 OOM（顯存不夠） | 可以在 `test_lora.py` 啟用 DiffSynth 官方的顯存管理／CPU offload（數值不變、只是變慢，記下用了什麼）；**不要**改解析度、幀數或模型。還是不行就把 `nvidia-smi` 峰值和日誌最後 50 行傳給我 |
| 校準的 `S3_RR_PSNR` 跟 27.2658 差超過 0.3 | 先停，把 `logs/` 整包傳給我，不要往下跑 |
| 編譯失敗（simple-knn、pointops2、rasterizer） | 把 `env/build_env.sh` 的完整輸出傳給我；常見原因是沒有 CUDA 11.8 的 nvcc（腳本會用 conda 裝） |
| 下載中斷 | 重跑同一個腳本即可（會續傳） |
| 跑到一半機器重開 | 重跑同一行，會從斷點續跑 |

任何時候都**不要**刪除或覆蓋已完成的 `OUT/` 內容。

---

## 5. repo 內容

| 檔案 | 用途 |
|---|---|
| `CLAUDE.md`、`AGENTS.md` | 給 AI 代理（Claude Code 等）的交付說明：目標、步驟、自我檢查、停下條件 |
| `setup_repo.sh` | 下載並核對官方程式與資料、套 patch |
| `env/build_env.sh` | 建 conda 環境；`env/` 另附官方 environment.yml、requirements.txt、README 原樣 |
| `fetch_weights.sh` | 下載 Wan2.1-I2V-14B-720P 與 4DSloMo LoRA 權重 |
| `run_all.sh` | 整條流程（stage-1 → 渲染 → 擴散 → stage-2 → 渲染 cam00），可續跑 |
| `collect_results.sh` | 打包要傳回的結果並算 md5 |
| `code/patches/`、`code/CHANGES.md` | 對官方程式的兩個小改動與理由（不改方法） |
| `code/OFFICIAL_CODE_MANIFEST.md5` | 官方程式 898 個檔案的 md5，用來確認下載到的是同一份 |
| `tools/` | 資料切段、組 stage-2 資料、渲染、評估的小工具 |
| `EXPECTED_OUTPUTS.md` | 每個輸出檔是什麼、怎麼檢查 |
| `DECISIONS_NEEDED.md` | 各個設定選擇的理由（參考用，跑的人不需要讀） |
| `slurm_example.sh` | Slurm 排程範本（不用 Slurm 就忽略） |

---

## 6. 來源與授權

- **4DSloMo 官方程式**：https://github.com/OpenImagingLab/4DSloMo （commit `9f75dd56f34a3440d3333bc048b9e2d5b4497cf3`）。本 repo **不重新散布**官方程式，`setup_repo.sh` 會從官方 GitHub 直接下載；授權見官方 repo 的 LICENSE。
- **模型權重**：`Wan-AI/Wan2.1-I2V-14B-720P`、`yutian05/4DSloMo`（HuggingFace 官方頁面），依各自授權。
- **資料**（Release 的 `data_slomo_k12_T.tar`）：取自 Neural 3D Video（N3DV）資料集（Li et al., CVPR 2022, https://github.com/facebookresearch/Neural_3D_Video ）的 `flame_steak`、`cook_spinach` 兩個場景，依 K=12 錯開快門排程抽出的影格＋相機參數＋初始點雲。原資料授權為 CC BY-NC 4.0，**僅供非商業研究使用**。
