# EXPECTED_OUTPUTS.md

`OUT/<scene>_<arm>/`（scene_arm ∈ flame_steak_KMOD12T, cook_spinach_KMOD12T[, flame_steak_OPT12T, cook_spinach_OPT12T 選配]；校準寫在 `OUT_calib/`）

| 路徑 | 內容 | 預期數量/尺寸 |
|---|---|---|
| `stage1/` | official train.py 的 model_path：`chkpnt7000.pth`、`cfg_args`、`default.yaml.used`、`input.ply`、`cameras.json`、point_cloud/ | 7000 it |
| `cam00_stage1/frame_0100.png … frame_0159.png` + `render_index.tsv` | stage-1 模型對 held-out cam00 的渲染（白背景） | 60 張 1352x1014 RGB |
| `views_src/` | 渲染訓練視角用的 render-only 資料夾（transforms_test.json = 20 台 x 60 幀，無真值） | 1200 筆 |
| `renders_train/cam01_0100.png … cam20_0159.png` | stage-1 對 20 台訓練相機、幀 100..159 的渲染（綠背景） | 1200 張 1352x1014 |
| `diffusion/chunkA/`、`chunkB/` | `in/test/ours_None/{renders,gt,video_crop,gt_crop}`（官方 process_video 輸出）、`out/images/<cam>_<0000..0032>.png`（擴散輸出）、`plan.json`、`cams.txt` | 每段 20 台 x 33 = 660 張 1024x1024 |
| `stage2_data/` | stage-2 資料集：`images/`（符號連結）、`transforms_train.json`（預設 drop0：20 x 59 = 1180 筆；keep0 1200；both 1320；+add_real 每台 +5）、`transforms_test.json`（cam00 x 60）、`points3d.ply` | 標頭 w=h=1024, fl_x=738.06, fl_y=738.24, cx=512.5, cy=512.0 |
| `stage2/` | 第二階段模型（同 stage1 的檔案結構） | 7000 it |
| `cam00_stage2/frame_0100.png … frame_0159.png` + `render_index.tsv` | **最終結果**：stage-2 模型對 held-out cam00 的渲染（白背景），命名與解析度和我們的真值一致 | 60 張 1352x1014 RGB |
| `psnr_stage1.tsv`、`psnr_stage2.tsv` | 每幀 PSNR（整幅 / 中央 [169,1183) 欄），僅供快速檢查 | 60 列 |
| `logs/<stage>.log`、`.done_<stage>` | 每階段日誌與完成標記（內有 wall_s） | |
| `prompt.txt` | 實際使用的 prompt | |

日誌中的 `S3_RR_PSNR <值>`（stage1_cam00.log、stage2_cam00.log）= render_s3_v2.py 對 60 幀整幅 PSNR 平均（train.py training_report 的定義）。
**stage-1 重現檢查**：KMOD12T 舊結果 flame_steak 27.2658、cook_spinach 27.5276（來源 （內部路徑，已略））；新環境/新 GPU 的 stage-1 應在 ±0.3 dB 內（3DGS 系訓練有隨機性，種子固定為 6667 但 CUDA 非決定性）。OPT12T 沒有舊值。

`OUT/preflight.txt`、`OUT/run_all.log`；`collect_results.sh` 產生 `OUT/<sa>_results.tar` + `.md5`（要寄回的東西）。
