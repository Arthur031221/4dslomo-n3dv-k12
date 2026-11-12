# 追加說明 c（11-12）：兩台機器平行跑追加包的 6 個場景

用同一個 repo、同一個 commit 就好，不用開新 repo。這份只改「誰跑哪幾個場景」；流程、設定、步數都和 `README_b.md` 相同。

## 1. 分工（各自照順序；每跑完一個就先回傳那一個）

| 機器 | 依序跑 |
|---|---|
| A：48 GB（舊工作包，已建好環境、已校準） | `sear_steak_KMOD12T` → `cut_roasted_beef_KMOD12T` → `coffee_martini_KMOD12T` |
| B：H200（新機器） | `flame_salmon_1_KMOD12T` → `flame_steak_HRV12T` → `cook_spinach_HRV12T` |

## 2. 機器 A
照 `README_b.md` 第 2、3 節做，只是場景換成上表 A 的三個：
```bash
for sa in sear_steak_KMOD12T cut_roasted_beef_KMOD12T coffee_martini_KMOD12T; do STEPS=50 bash run_all_b.sh $sa; bash collect_results.sh $sa; done
```

## 3. 機器 B（H200 = sm_90）
1. `git clone` 本 repo，照 `README.md` 第 2 節從頭做一次：`md5sum -c MD5SUMS`、`setup_repo.sh`（會下載 Release v1 的舊資料，校準要用）、`env/build_env.sh`、`fetch_weights.sh`、校準（`OUT_calib`，三項都通過才 `touch CALIBRATED`）。
   唯一的差別：**建環境之前、以及之後每次執行之前，都先 `export TORCH_CUDA_ARCH_LIST=9.0`**。CUDA 擴充要替 H200 重新編譯；`env/build_env.sh` 已改成讀這個變數（沒設時仍是 8.9）。
   - A、B 之間如果能直接傳檔，可以把 A 的 `work/checkpoints/`（權重）複製到 B，省下載時間，之後仍要跑 `fetch_weights.sh` 核對（已存在的會 SKIP）。
   - 不要從 A 複製 conda 環境、`work/torch_ext/` 或 `CALIBRATED`：這三樣在 B 上自己建、自己校準。
2. 從 Release v2（tag `v2-data-b`）下載 `data_slomo_k12_T_b.tar` 和 `MD5SUMS_big_b.txt`，`md5sum -c MD5SUMS_big_b.txt` 通過後，照 `README_b.md` 第 2 節解到 B 的 `work/data`。
3. 跑：
```bash
export TORCH_CUDA_ARCH_LIST=9.0
for sa in flame_salmon_1_KMOD12T flame_steak_HRV12T cook_spinach_HRV12T; do STEPS=50 bash run_all_b.sh $sa; bash collect_results.sh $sa; done
```

## 4. 兩台都不能改的
同一個 commit、`default.yaml`、seed 0、擴散 50 步、同一份權重。除了 B 的 `TORCH_CUDA_ARCH_LIST=9.0`，其他設定一律不改。

## 5. 每個場景回傳時附上
- GPU 型號：`nvidia-smi --query-gpu=name --format=csv,noheader`
- torch／CUDA 版本：`python -c "import torch; print(torch.__version__, torch.version.cuda)"`
- stage-1 讀數：`grep S3_RR_PSNR OUT/<場景>/logs/stage1_cam00.log`
- `OUT/<場景>_results.tar` 和它的 `.md5`

## 6. 時限
台北時間 11-15 24:00 之前回傳的才用得上；之後才完成的不用再跑。
