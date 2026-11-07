# DECISIONS_NEEDED.md -- 每一個我必須做選擇的地方（不是默默猜的）

格式：**決定**（已實作的選項）／備選／官方程式碼依據（檔:行號，官方 commit 9f75dd56）。
先講「官方程式到底怎麼吃擴散輸出」（#0），因為後面的選擇都由它推出。

## #0 官方 stage 2 / FixModel 的實際行為（逐行讀出來的）
- **擴散條件 = 「真值第 0 幀」＋「整段渲染影片」**。`test_lora.py` l.77：`gt_crop/<cam>_0000.png` 當 `input_image`；l.85-101：`video_crop/<cam>.mp4` 讀成 `input_video`。
  `FixModel.py` l.266-272 -> `encode_image_condition`（l.174-193）：CLIP 特徵取自 input_image（l.175-176）；VAE 輸入 `vae_input`（l.183）第 0 幀 = input_image、其餘補 0，
  然後 l.185-187 `for idx in range(1, len(inter_images)): vae_input[:, idx] = inter_images[idx]` -> **渲染影片的第 1..N-1 幀逐幀塞進 VAE 輸入；渲染影片的第 0 幀被丟掉**（由真值取代）。
  遮罩 `msk`（l.178）全為 1，**沒有**像一般 I2V `encode_image`（l.159 `msk[:,1:] = 0`）那樣把後面幀標成「待生成」，也就是整段影片都被當成「已給定」條件，DiT 做的應是「修補」而非「續寫」（此句為我的推論，不是程式碼明文）。
- 輸出：`save_img`（test_lora.py l.11-21）寫 `<output>/images/<cam>_<i:04d>.png`，i = 0..32，1024x1024（l.114 `height=1024,width=1024`）。
- **stage 2 資料 = 只有擴散輸出，不含原始拍到的影格**。README l.72-75：`dance_demo10_wan/` 只有 `test_lora` 產生的 `images/`，加上從 `dance_demo10/` 複製來的 `transforms_train_stage2.json`（改名 transforms_train.json）、`transforms_test_demo.json`（改名 transforms_test.json）、`points3d.ply`。
  `transforms_train_stage2.json`：標頭 `w=h=1024, fl_x=fl_y=1256.70, cx=cy=512`；396 筆 = 12 相機 x idx 0..32；`file_path='images/<cam>_<idx:04d>'`；`time = idx/36`（對所有相機相同，不管該相機實際拍攝相位）。
  標頭數字 = 原圖 2048x2448 經 process_video 縮放（x0.5）再中央裁切後的內參：1256.70 = 2513.40x0.5，cy = 1224x0.5 - 100 = 512。`tools/slomo_common.py:stage2_header` 用同一公式（對 1352x1014 算得 fl_x=738.06, fl_y=738.24, cx=512.5, cy=512.0）。
- stage 2 訓練：同一支 `train.py`、同一個 `configs/default.yaml`（README l.76 的指令與 stage 1 只差路徑），因此 7000 it、batch 4、從 `points3d.ply` 重新初始化（**不是**從 stage-1 checkpoint 接著訓練）。
- 官方 demo 的 stage-2 測試相機（transforms_test_demo.json）是訓練相機之一（19305309）在沒拍到的時刻，不是空間 hold-out。我們的協定是 cam00 空間 hold-out，所以 cam00 **完全不進擴散**（它的真值第 0 幀不會被用）。
- 其他寫死 dance 的地方：test_lora.py l.70 相機清單、l.108 prompt、l.113 seed/num_frames；render.py l.66-68 綠背景；`process_video.py` 以 `split('_')[0]` 當相機名。其餘（FixModel、train、scene）與資料集無關。

## #1 分段（33 幀上限 vs 我們的 60 幀視窗）
**決定**：兩段，A = 窗口幀 100..132，B = 窗口幀 127..159（重疊 127..132 共 6 幀），各 33 幀，每台訓練相機各跑一次。
備選：(a) 一段 61 幀（`--num_frames 61`，FixModel l.243-245 會把幀數捨入成 4n+1）：token 數 ×1.8、注意力成本 ×3.3，且 LoRA 很可能只在 33 幀上訓練，屬分佈外，風險大；(b) 100..132 + 133..159(27 幀) 不重疊：接縫處沒有雙方向的真值錨點；(c) 3 段（100-132/114-146/127-159）：算力 ×1.5；(d) 每台相機的段起點對齊它自己拍到的第一幀（類似 demo 的「相位」）：每台相機的段終點不同，B 段會超出 159（模型 time_duration 只到 10.0＝窗口末尾），不可行。
依據：test_lora.py l.113 `num_frames=33`、README l.63 `--max_frames 33`、process_video.py `--max_frames`。

## #2 重疊幀歸屬與 idx 0
**決定（ASSIGN=drop0，預設）**：丟掉每段輸出的 idx 0；A 的 idx 1..27 -> 幀 101..127，B 的 idx 1..32 -> 幀 128..159。每台相機 59 張擴散影像（幀 100 沒有）。
理由：idx 0 是在重畫「真值第 0 幀」，而該真值影像的拍攝時刻與段起點可能差很多（#3：A 段最多差 ~11 幀），把它標在格點時間會是錯誤的時間標籤；demo 是 4 倍幀率、差 <=3 幀 (1/36 s) 因而沒問題，我們是 12 倍。重疊區 127..132 取 B 的前段（距離真值錨點近）而不是 A 的尾段（累積漂移最大）。
備選：`ASSIGN=keep0`（demo 逐字：A idx0..26 -> 100..126，B idx0..32 -> 127..159，每相機 60 張，含兩張時間標錯的 idx0）；`ASSIGN=both`（兩段的所有輸出都用，重疊區每幀兩張，B 的檔名加 b；1320 張）。
依據：stage2.json 對每段 idx 0..32 一律 time=idx/36（#0）。

## #3 「frame 0」對交錯快門相機是什麼
**決定**：段起點 s（A=100、B=127）；真值第 0 幀 = 該相機真實拍到的影格中「真實拍攝時間（slot + res）最接近 s」者（平手取較早），來源是 stage-1 資料的 transforms_train.json 的 `frame`/`res`；`prep_chunks.py` 把選擇與時間差記在 `diffusion/chunk{A,B}/plan.json`。渲染影片仍以格點時間 s..s+32 對齊（`render_views.py` 用 time = 10*(f-100)/60）。
K=12：每台相機每 12 幀才拍一張，所以真值錨點與段起點最多差 11 幀（A）/ 5.6 幀（B）（用 flame_steak_OPT12T 的 keep 實測：A 段最大 |差| = 11.02 幀，B 段 5.55 幀）。這和 demo 的行為同型（demo：相位 p 的相機以它第一張拍到的影像當第 0 幀，卻把它當 t=0），但我們的差距更大。
備選：每台相機把段起點移到它自己的錨點（不可行，見 #1(d)）；把 GT 幀也補在對的時間（`--add_real`，見 #6，這個已實作成旗標）。

## #4 Prompt
**決定**：每場景一句中性描述，可用 `PROMPT=` 覆寫：flame_steak -> "A person cooking a steak in a kitchen"；cook_spinach -> "A person cooking spinach in a kitchen"。不描述動作、不提火焰細節（避免替模型補內容）。
備選：官方 "A girl dancing"（內容不符）；空字串（`cfg_scale=5.0` 的 CFG 對空 prompt 退化，FixModel l.261-262）。negative prompt 維持官方原文（test_lora.py l.109）。
我無法驗證 prompt 對輸出的影響；建議校準時（README #校準）順便用兩種 prompt 各跑 1 台相機看差異。

## #5 解析度、裁切、內參
**決定**：完全照官方：原圖 1352x1014 -> 短邊縮放到 1024（`int(1352*(1024/1014)) = 1365` 寬）-> 中央裁切 1024x1024（x0 = 170）（process_video.py l.27-42）。stage-2 內參 `fl_x=738.06, fl_y=738.24, cx=512.5, cy=512.0`（推導見 #0）。
**後果（必讀）**：stage-2 模型只被監督到原圖中央約 1014x1014 的視野（原圖 x∈[168.4,1183.1)），左右各 ~168 px 沒有任何監督；held-out cam00 以完整 1352x1014 渲染時，左右邊緣是未被擬合的區域。官方流程本來就是這樣（它的 dance demo 同樣裁掉 2048x2448 的上下各 100 px，但裁得少很多）。
因此 `eval_cam00.py` 同時報整幅 PSNR 與中央 [169,1183) PSNR；我們自己會用兩種方式算。備選：把原圖補黑邊/縮小成 1024x768 再補成 1024（改 process_video，偏離官方，LoRA 是否能處理未知）；分兩個 1014x1014 的裁切（左/右）各跑一次擴散（算力 x2，且需另做 stage-2 內參，我沒有實作）。

## #6 stage-2 資料組成
**決定**：只用擴散輸出（官方 demo 做法，#0）。`ADD_REAL=1` 會額外加入真實拍到的影格（同樣 Lanczos 縮放+裁切，標真實時間 `time`），每台相機 +5 張。
備選：ADD_REAL=1 是否更好我不知道；主跑請用預設，ADD_REAL=1 若算力有餘可當對照。

## #7 迭代數與 config
**決定**：stage 1、stage 2 皆 `configs/default.yaml` 原樣（7000 it，README l.54/l.76 的指令）。stage 2 的影像數（約 1180）是 stage 1（100）的十倍多，7000 it 是否夠我不確定，但改 config 就偏離官方；若要加 iterations，請另開 run（不要覆蓋）。

## #8 背景色
**決定**：餵給擴散的訓練視角渲染用官方 render.py 的綠背景 [0.125,0.216,0.157]（`--render_bg green`；LoRA 是在這種輸入上訓練/示範的，推論）；held-out cam00 的評估渲染用白背景＝訓練背景（和舊 stage-1 評估一致，N3DV 影像無 alpha，兩種背景在不透明區域無差別）。
備選：全白。N3DV 場景幾乎全不透明，影響應很小；我無法驗證。

## #9 資料與時間標籤
**決定**：用舊 session 已建好的 `*_OPT12T` / `*_KMOD12T` 資料夾原樣（tar -h 解參考，640 張 PNG）。這些是「T 版」＝ 與主表 STG 臂完全相同的影格與**真實拍攝時間標籤**（`time = 10*(f-100+res_c)/60`，res_c 為各相機相對整數格點的子幀偏移，|res|<=0.5；影像取自 realized tree 的原始幀 f - r_c；cook_spinach 的來源幀含 98、99）。因此 stage 1 與舊 4DSloMo-noDiff 的輸入逐位元一致，stage 1 可直接對舊結果（KMOD12T：flame 27.27 dB、cook 27.53 dB，S3_RR_PSNR）做重現檢查；OPT12T 的 stage-1 舊結果 104 上沒有 run 目錄，所以無參考值。
擴散階段的渲染影片使用無偏移的格點時間（time = 10*(f-100)/60），因為輸出要重新當成「同步格點上的虛擬影格」使用。
備選：從原始 N3DV 幀重建（`builders_reference/mk_slomo_data_t3.py`，需要 104 上的 realized tree 與 toff 檔，遠端沒有）。

## #10 擴散步數、cfg、seed
**決定**：`STEPS=50`（官方預設 test_lora.py l.28；README l.66 說 5 步約達 80% 品質）。cfg_scale=5.0（FixModel l.230 預設）、seed=0（官方）、tiled VAE（官方）皆不動。
算力吃緊時請用 `STEPS=5` 先跑主臂，並在回報中註明；50 步的結果才算「官方完整設定」。

## #11 官方環境與編譯
**決定**：python 3.10 + torch 2.4.1+cu118（README l.38-42）；`env/environment.yml` 是舊 4DGS 檔（python 3.7/torch 1.12），README 不用，不採用。額外加的只有 `numpy<2, ninja, opencv-python-headless`（官方 requirements.txt 沒列 cv2，但 process_video.py/test_lora.py 都 import cv2；diffsynth 可能已經帶入）。`pip install ./simple-knn ./pointops2` 加 `--no-build-isolation`（它們的 setup.py 要 import torch）。TORCH_CUDA_ARCH_LIST=8.9（RTX 6000 Ada）。
舊 session 是在 python 3.11 + CUDA 12.1 的環境跑 stage 1 的（104），檢查 stage-1 PSNR 是否重現可抓出環境造成的差異。

## 我無法驗證的事（沒有網路、沒有 GPU）
1. `yutian05/4DSloMo` 裡的權重檔名是否正是 `4DSloMo_LoRA.ckpt`（test_lora.py l.27 預設路徑如此；fetch_weights.sh 失敗時會列出 repo 檔案清單一次）。
2. `google/umt5-xxl/` 的四個 tokenizer 檔名（FixModel.py l.131 要求 `checkpoints/google/umt5-xxl`；檔名是依 Wan2.1 HF repo 慣例寫的）。
3. 7 個 DiT shard 的總大小（我估 ~65 GB fp32 儲存）、CPU RAM 峰值（估 >= 64 GB）、單卡 48 GB 是否夠 14B bf16（約 28 GB）+ 1024x1024x33 幀 的啟動記憶體。
4. 所有 GPU 時數（見 README，來自 FLOPs 推算，不是量測）。
5. 整條流程沒有在任何機器上端到端跑過；只做了：語法檢查、`run_all.sh` 控制流用假的 python 取代 GPU 程式的流程測試、資料工具用真實 json 的邏輯測試。


## 11-06 14:0x 定案（v2 包）
- 主要臂改 KMOD12T（主表所有錯開類方法用同一組 KMOD12 輸入；舊 stage-1 也是 KMOD12T，重現基準 flame 27.2658 / cook 27.5276）。OPT12T 降為選配。
- 校準必做（MAX_CAMS=1 STEPS=5，OUT_calib），run_all.sh 在沒有 CALIBRATED 檔時拒絕正式跑。
- #2 ASSIGN=drop0（對對手較有利：idx 0 是另一個時刻的真實影格，keep0 會把錯時刻的影像標成區段起點）。
- #3 GT 首幀 = 真實拍攝時間最接近區段起點的影格（官方流程要求首幀是真實影像；用渲染當首幀會把瑕疵當乾淨錨點）。
- #4 中性場景提示詞；#5 官方 1024 中央裁切（我方評估同時報整幅與中央）；#6 只用擴散輸出（官方）；#7 default.yaml 原樣；#10 STEPS=50、cfg 5.0、seed 0。
- 步數：正式 50 步（官方預設值）；若 NCHC 時數不夠，先交 5 步版（官方 README 範例指令就是 5 步），回報時標明。
