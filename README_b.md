# slomo_full_k12_b -- 追加工作包：再多跑幾個場景的 4DSloMo 完整版（流程和先前完全相同）

這是「追加包」，不是新包。環境、權重、repo、校準都沿用你們機器上已經建好的那一份，**不需要重做校準、不需要重建環境、不需要重下載權重**。
唯一的新東西：(1) 一個新的資料 tar（6 個新的場景資料夾）；(2) 一支 `run_all_b.sh`（= 舊 `run_all.sh` 複本，只改了「允許的場景名單」、幾個寫死「20 台相機」的檢查，以及替四個新場景各加一句 prompt，改動全部列在 `run_all_b.diff`）。
流程、設定檔（default.yaml）、seed、擴散步數（50）、其他所有腳本（`tools/`、`setup_repo.sh`、`collect_results.sh`、`fetch_weights.sh`）都沿用舊檔、一個字都沒改。

## 1. 要做什麼（6 個項目，每個獨立，約 15–20 小時一個）
場景名稱 `<場景>_<排程>`：KMOD12T 是 4DSloMo 自己的交錯快門排程，HRV12T 是我們的排程；相機位置、測試窗（第 100–159 幀）、cam00 當 held-out 都一樣。
**請照這個順序跑，一個跑完就先回傳那一個，不要等全部：**

| 順序 | SCENE_ARM | 訓練相機數 | 資料夾內 images/ 張數 |
|---|---|---|---|
| ① | `sear_steak_KMOD12T` | 20 | 160 |
| ② | `flame_steak_HRV12T` | 20 | 160 |
| ③ | `cut_roasted_beef_KMOD12T` | 19 | 155 |
| ④ | `cook_spinach_HRV12T` | 20 | 160 |
| ⑤ | `coffee_martini_KMOD12T` | 17 | 145 |
| ⑥ | `flame_salmon_1_KMOD12T` | 18 | 150 |

- **時限：台北時間 11-15 24:00 之前回傳的才用得上；之後才完成的不必再跑。** 以每景約 19 小時（相機少的場景會略快）計，大約能完成前 4～5 個；來不及跑的就不跑，不要為了趕時間改步數（要改請先回報，並在回報裡標明）。
- ⑤⑥ 的 stage-1 在我們 32 GB 的卡上曾經記憶體不足（峰值約 32–38 GB），你們的 48 GB 卡可以。
- 如果只能做一部分，優先順序就是上表的順序。

## 2. 取得並解開新資料 tar（只做一次）
檔案（我方放在你們收件處，跟先前的大檔同一個方式）：`data_slomo_k12_T_b.tar`，約 1.22 GB（1,217,239,040 bytes），md5 = `51f36df5f32677c2122a678235864a97`（同目錄 `MD5SUMS_big_b.txt`）。
```bash
cd <放 tar 的資料夾>
md5sum -c MD5SUMS_big_b.txt        # 要顯示 data_slomo_k12_T_b.tar: OK 才繼續
# 解到舊工作包的 work/data（和舊 tar 解的是同一個資料夾；新名稱不會和舊的衝突）
tar -xf data_slomo_k12_T_b.tar -C <舊工作包>/work/data
for d in sear_steak_KMOD12T flame_steak_HRV12T cut_roasted_beef_KMOD12T cook_spinach_HRV12T coffee_martini_KMOD12T flame_salmon_1_KMOD12T; do
  echo "$d images: $(ls <舊工作包>/work/data/$d/images | wc -l)"; done     # 要看到 160 160 155 160 145 150
```
（這個 tar 內的影像都是實體檔，沒有符號連結；每個資料夾有 images/、transforms_train.json、transforms_test.json、points3d.ply、keep.json、meta.json。）

## 3. 怎麼跑（在「已經建好環境並校準過的那個舊工作包資料夾」裡）
```bash
cd <舊工作包>                       # 就是放 run_all.sh、CALIBRATED、work/、OUT/ 的那個資料夾
cp <新包>/run_all_b.sh .            # 必須放在同一個資料夾（它用自己所在的資料夾找 tools/、work/、OUT/、CALIBRATED）
# 照你們上次的方式先啟用環境（local/full.sh 做的事：啟用環境、背景開顯存取樣），然後一次只跑一個項目：
STEPS=50 bash run_all_b.sh sear_steak_KMOD12T
bash collect_results.sh sear_steak_KMOD12T      # 舊的 collect_results.sh 直接可用，產生 OUT/sear_steak_KMOD12T_results.tar 與 .md5
```
- `CALIBRATED` 檔在舊資料夾已存在，所以不會被「校準防護」擋下；若不存在才需要補（先前已決定接受校準結果）。
- 想一次排完整串也可以（每個跑完自動打包），例如：
  `for sa in sear_steak_KMOD12T flame_steak_HRV12T cut_roasted_beef_KMOD12T cook_spinach_HRV12T coffee_martini_KMOD12T flame_salmon_1_KMOD12T; do STEPS=50 bash run_all_b.sh $sa; bash collect_results.sh $sa; done`
  但請每個 tar 一做好就回傳（不用等整串結束）。
- 可以續跑：被中斷後用同樣的指令重跑，已完成的階段會自動跳過（`OUT/<sa>/.done_*`）。**不要覆蓋或刪除 `OUT/` 內已完成的內容。**
- 換場景時 `OUT/preflight.txt` 和 `OUT/run_all.log` 會被更新/附加（舊場景的結果 tar 不受影響）。
- 磁碟：每個項目輸出約 15–20 GB（17–20 台相機，台數少的較小），再加舊的結果；回傳完一個後如需空間，可把已回傳的 `_results.tar` 移走，但請保留 `OUT/<sa>/` 內容直到我方確認收到。

## 4. 這個追加包和舊包「相同」與「不同」
- 相同：官方 4DSloMo（同一個 repo 與 commit）、`configs/default.yaml`（md5 3b8355b189b7dbd064fdec76033bf7c8）、stage-1 7000 it、50 步擴散、seed 0、ASSIGN=drop0、ADD_REAL=0、不設 PROMPT 環境變數（用 `prompt_of` 的每景句子）。
- 不同（只有這些，詳見 `run_all_b.diff`）：
  1. 允許的 SCENE_ARM 名單多了 6 個新名稱，預設清單改為這 6 個（也可以像上面那樣一次只傳一個名稱）。
  2. 原本寫死的數量改成依場景自動算：資料夾 images/ 張數（160 → 145／150／155／160）、每段擴散輸出張數（660 = 20 台 x 33 → 相機數 x 33）、訓練視角渲染張數（1200 → 相機數 x 60）、處理後影片數（20 → 相機數）。
- prompt：舊規則只有 `flame_steak*`、`cook_spinach*` 有專屬句子。追加包照同一個句型替四個新場景各加一句（`run_all_b.sh` 的 `prompt_of`）：sear_steak＝"A person cooking a steak in a kitchen"（與 flame_steak 相同）、cut_roasted_beef＝"A person cutting roasted beef in a kitchen"、coffee_martini＝"A person making a coffee martini in a kitchen"、flame_salmon_1＝"A person cooking salmon in a kitchen"。flame_steak、cook_spinach 的句子不變。實際用的句子會寫在 `OUT/<sa>/prompt.txt`。
- stage-1 的重現參考值在 `EXPECTED_OUTPUTS_b.md`。先前的經驗：你們機器上 flame_steak 的 stage-1 比我方高約 0.4 dB，已被接受；所以這次**差 >0.3 dB 不用停下，照跑，只要在回報裡寫出實際值**。若差 >2 dB（或訓練根本沒收斂）請先回報再繼續。

## 5. 請寄回什麼（和舊包相同）
- 每個項目的 `OUT/<SCENE_ARM>_results.tar` 與 `OUT/<SCENE_ARM>_results.tar.md5`（寄回前先 `md5sum -c` 自檢）。
- 另附：最新的 `OUT/run_all.log`；回報裡寫每個項目的 stage-1 `S3_RR_PSNR`（`OUT/<sa>/logs/stage1_cam00.log`）、stage-2 `S3_RR_PSNR`、各階段秒數（`.done_*` 內的 wall_s）、顯存峰值，以及任何 `[FAIL]`／`[PART]`。
- 其他環境檔（preflight、pip_freeze、fetch_weights.log）已經寄回過，環境沒變就不用再寄。
