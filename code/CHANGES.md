# CHANGES.md -- 對官方 4DSloMo (commit 9f75dd56f34a3440d3333bc048b9e2d5b4497cf3) 的所有改動

官方原始碼：本地唯讀副本 （內部路徑，已略）
（GitHub 該 commit 的封存，無 .git）。tarball：`4DSloMo-9f75dd56.tar.gz`（內容 = 該資料夾原樣，md5 見 MD5SUMS_big.txt）。
下列 patch 用 `patch -p1`（在 repo 根目錄）套用，由 setup_repo.sh 自動做。**沒有任何一處改 loss、模型、config、超參數。**
`configs/default.yaml` 不動（md5 3b8355b189b7dbd064fdec76033bf7c8，與舊 session 全部 stage-1 跑次相同；run_all.sh 開跑前檢查）。

## 0001_stage1_identical_to_old_runs.diff（4 個 hunk，3 個檔）
這就是舊 session 跑所有 stage-1 實驗時用的程式（（內部路徑，已略） 相對官方的唯一差異；已用 md5 逐檔核對，
其餘 .py/.cu/.cpp/.yaml 與官方逐位元相同）。
| 檔案:hunk | 內容 | 為什麼 |
|---|---|---|
| utils/data_utils.py (1 行) | `dtype=np.byte` -> `np.uint8` | `arr*255.0` 轉成有號 int8 會溢位（值 >127 在新版 numpy/PIL 下行為未定義或報錯）；uint8 才是 RGB 影像的正確型別，數值上不改變訓練目標 |
| scene/dataset_readers.py (2 行) | 同上，兩處 `np.byte` -> `np.uint8` | 同上 |
| train.py (hunk 1) | 在 `scene.save(iteration)` 後，若 `iteration == opt.iterations` 另存 `chkpnt<iter>.pth` | 舊 session 為確保 7000 次的最終 checkpoint 存在而加；官方 `Scene.save` 其實已寫同名檔 `chkpnt<iter>.pth`，所以此 hunk 是冗餘的純存檔，不影響訓練。為了和舊 stage-1 完全同碼而保留 |
| train.py (hunk 2) | 解析完 yaml 之後再 `args.save_iterations.append(args.iterations)` | 官方在 yaml 合併之前 append，yaml 若改 iterations 則最終 iteration 不在 save 清單；純存檔邏輯 |

## 0002_test_lora_parameterize.diff（test_lora.py，5 個 hunk）
官方把 dance demo 的常數寫死。預設值全部等於官方寫死的值（不傳參數 = 官方行為）。
| 位置（官方行號） | 改動 | 為什麼 |
|---|---|---|
| l.28 之後 | 新參數 `--prompt`（預設 "A girl dancing"）、`--cams`（逗號清單或 `@檔案`；預設 dance 12 台）、`--num_frames`（預設 33）、`--seed`（預設 0）、`--skip_existing` | l.70 `cam_list`、l.108 `prompt`、l.113 `seed=0,num_frames=33` 皆寫死為 dance demo；N3DV 的相機名是 cam01..cam20 |
| l.70 | `cam_list` 由 `--cams` 解析 | 同上 |
| l.75 之後 | `--skip_existing`：該相機輸出 `images/<cam>_<i:04d>.png` 已齊全就跳過 | 一次要跑 20 台×2 段，可續跑；不改任何推論數值 |
| l.108 | `prompt=args.prompt` | 見 DECISIONS_NEEDED.md #4 |
| l.113 | `seed=args.seed, num_frames=args.num_frames` | 同上 |

## 沒有 patch 的官方檔案（有意不改，原因）
- `render.py`：**不使用**。它建 `GaussianModel(sh_degree, gaussian_dim=4, rot_4d=True)` 時沒帶訓練 config（time_duration / sh_degree_t / eval_shfs_4d），
  舊 session 實測同一個 7000 it checkpoint 用它渲染只有 24.63 dB，而 train.py 自己的 test 評估是 30.60 dB（render_s3_v2.py 檔頭有記載）；
  另外它在 l.66-68 把背景寫死成綠色 [0.125,0.216,0.157]。改用 `tools/render_views.py`（訓練視角，綠背景＝官方 render.py 的預設）與 `tools/render_s3_v2.py`（held-out cam00，白背景＝訓練背景）。兩者是新檔、不是 patch。
- `process_video.py`：原樣使用（它以檔名 `split('_')[0]` 當相機名，`cam01_000.png` 之類的檔名正好可用）。
- `FixModel.py`、`train.py` 其餘部分、`scene/*`、`gaussian_renderer/*`、`configs/default.yaml`：原樣。
- `diff-gaussian-rasterization/setup.py` 不可用（packages=['diff_gaussian_rasterization'] 但資料夾不存在）；官方 `gaussian_renderer/diff_gaussian_rasterization.py` l.15-25 在 import 時用 `torch.utils.cpp_extension.load` JIT 編譯，所以不需要（也不要）pip 安裝它。

## 新增檔案（tools/，不在官方 repo 內）
slomo_common.py、mk_views.py、prep_chunks.py、assemble_stage2.py、eval_cam00.py、render_views.py、render_s3_v2.py（後者 = 舊 stage-1 用的 render，md5 73e22544b542ce26772dc91e70e9e17c，逐位元複製）。
setup_repo.sh 會把 render_views.py、render_s3_v2.py 複製到 repo 根目錄（它們 import repo 內的套件）。
