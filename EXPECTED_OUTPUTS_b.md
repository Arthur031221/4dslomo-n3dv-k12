# EXPECTED_OUTPUTS_b.md（追加包）

輸出資料夾結構、檔名、日誌與 `S3_RR_PSNR` 的定義和舊 `EXPECTED_OUTPUTS.md` 完全相同（`OUT/<scene>_<arm>/…`），只有「張數」隨相機數改變：

| SCENE_ARM | 訓練相機數 | data images/ | views_src 與 renders_train | 每段擴散 chunkA／chunkB | stage2_data 訓練張數（drop0，每台 59 張） |
|---|---|---|---|---|---|
| sear_steak_KMOD12T | 20 | 160 | 1200 | 660 | 1180 |
| flame_steak_HRV12T | 20 | 160 | 1200 | 660 | 1180 |
| cut_roasted_beef_KMOD12T | 19 | 155 | 1140 | 627 | 1121 |
| cook_spinach_HRV12T | 20 | 160 | 1200 | 660 | 1180 |
| coffee_martini_KMOD12T | 17 | 145 | 1020 | 561 | 1003 |
| flame_salmon_1_KMOD12T | 18 | 150 | 1080 | 594 | 1062 |

cam00_stage1／cam00_stage2：每項 60 張 1352x1014 RGB（frame_0100 … frame_0159）。

## stage-1 重現參考值（指標 = `S3_RR_PSNR`，render_s3_v2.py 對 60 幀整幅 PSNR 的平均；與舊包相同的定義）
來源是我方自己用同一份官方程式碼（4DSloMo commit 9f75dd56）、同一個 `default.yaml`（md5 3b8355b1）、7000 it、**同樣這 6 份資料夾**（見下方「資料來源」）跑的 stage-1，取自各 `*.render.log` 的 `S3_RR_PSNR` 行。
容許範圍同舊包 ±0.3 dB；超出不用停（見 README_b.md 第 4 節），只要在回報寫出實際值。

| SCENE_ARM | 參考 S3_RR_PSNR | 我方跑的機器 | 來源檔 |
|---|---|---|---|
| sear_steak_KMOD12T | 28.2609 | cvlab-104（job j07s1，顯存峰值 26.6 GB） | 104:/data/shared/kylechen/stagger27/logs/j07s1/s1_sear_steak.render.log |
| flame_steak_HRV12T | 29.4153 | cvlab-102 RTX 4090 24 GB（job j37c） | /cvlab-40t/kylechen/stagger27x/out/j37c/s1h_flame_steak/s1h_flame_steak.render.log |
| cut_roasted_beef_KMOD12T | 27.3783 | cvlab-102 RTX 4090 24 GB（job j37a） | /cvlab-40t/kylechen/stagger27x/out/j37a/s1_cut_roasted_beef/s1_cut_roasted_beef.render.log |
| cook_spinach_HRV12T | 29.2733 | cvlab-102 RTX 4090 24 GB（job j37c） | /cvlab-40t/kylechen/stagger27x/out/j37c/s1h_cook_spinach/s1h_cook_spinach.render.log |
| coffee_martini_KMOD12T | 21.1304 | cvlab-104（job j07r2，顯存峰值 38.3 GB） | 104:/data/shared/kylechen/stagger27/logs/j07r2/s1_coffee_martini.render.log |
| flame_salmon_1_KMOD12T | 22.5111 | cvlab-104（job j07s4，顯存峰值 32.4 GB） | 104:/data/shared/kylechen/stagger27/logs/j07s4/s1_flame_salmon_1.render.log |

（同一批 stage-1 跑次在我方主表用的另一個指標是「動態遮罩區 M」，不是 S3_RR_PSNR，這裡不用。）
stage-2（最終結果）沒有參考值，只需回報。

## 資料來源與正控制（我方留底）
- sear／coffee／salmon（KMOD12T）：直接取自 104:/data/shared/kylechen/stagger27/data/slomo_k12T/<scene>_KMOD12T（符號連結解參考後打包）。
- cut_roasted_beef_KMOD12T、flame_steak_HRV12T、cook_spinach_HRV12T：這三份原本只存在於 cvlab-102 的本機磁碟（104 沒有）。我方在 103 用同一支建資料程式（code/slomo102/mk_slomo_data_102.py，md5 7ab29ccf）從 104 的輸入重建（輸入 651 檔 md5 全與 104 相同），`--verify` 通過。
  建資料程式的正控制：以同程式重建 flame_steak_KMOD12T，對舊 tar 的來源資料夾（104:/data/shared/kylechen/S3/data_slomo/flame_steak_KMOD12T，舊 tar 內檔案與其 md5 逐檔相同）：160 張影像 md5 全同、points3d.ply md5 同（ed8578f9…）、keep.json 同，cmp_build.py 輸出 CMP_OK（transforms 浮點最大差 0）。以同程式重建 sear_steak_KMOD12T，對 104 的 j07s1 實際訓練資料夾：160 張影像 md5 全同、CMP_OK。
