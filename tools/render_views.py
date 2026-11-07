# render_views.py -- bundle slomo_full_k12. = render_s3_v2.py (md5 73e22544b542ce26772dc91e70e9e17c, the renderer of the old stage-1 runs) with 3 differences only:
#  (1) the view cameras are those of <source_path>/transforms_test.json (made by mk_views.py: training cameras x 60 frames, NO ground truth needed);
#      the (dummy) gt images are never read (scene.test_cameras is used directly instead of the CameraDataset),
#  (2) background colour is a parameter: --render_bg green (= official render.py l.68 hard-coded [0.125,0.216,0.157], what the diffusion LoRA was run with) | white (= training bg),
#  (3) output file = <out_dir>/<camXX_NNNN>.png (name of the view image), --skip_existing makes the run resumable.
# Everything else (training-config handling, GaussianModel / Scene calls, eval_shfs_4d, clamp to [0,1]) is the v2 text unchanged.
# render_s3_v2.py -- S3 W26 (10-31). render_s3.py (untouched) + the TRAINING config. Why: render_s3.py built GaussianModel(sh_degree, gaussian_dim=4, rot_4d=True)
# without time_duration / force_sh_3d / sh_degree_t, called Scene(...) without time_duration / num_pts, and left pipe.eval_shfs_4d at its default (False):
# a 7000-it checkpoint rendered at 24.63 dB (half transparent, blurry) while train.py's own test eval said 30.60 dB.
# v2 = same outputs as render_s3.py (white training background, OUT/frame_NNNN.png with NNNN = true frame number, OUT/render_index.tsv), but
#  1. --config <yaml> is read exactly like train.py (same parser groups/extras, recursive_merge copied verbatim, the yaml overrides args),
#     EXCEPT that model_path / source_path / loaded_pth given for rendering are re-applied after the merge (the training yaml has `loaded_pth: ""`).
#  2. frame_ratio / time_duration handling and the GaussianModel(...) / Scene(...) calls are those of train.py lines 39-46
#     (shuffle=False is kept from render_s3.py: it only changes the order of the test cameras).
#  3. --check_psnr: mean over getTestCameras() of psnr(clamp(render,0,1), gt).mean() (train.py training_report definition); prints "S3_RR_PSNR <v>".
# (safe_state appends " [dd/mm HH:MM:SS]" to every printed line: parse the value as the 2nd whitespace-separated token.)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import re, sys, torch, torchvision
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state
from utils.image_utils import psnr

if __name__ == "__main__":
    parser = ArgumentParser(description="S3 render v2 (render_s3.py + training config)")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)          # not used for rendering; only so that every key of the training yaml exists on args (as in train.py)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--check_psnr", action="store_true")   # kept for CLI compatibility; ignored here (no gt)
    parser.add_argument("--render_bg", type=str, default="green", choices=["green", "white"])
    parser.add_argument("--skip_existing", action="store_true")
    # extras of train.py's parser that the yaml sets (copied from train.py)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6667)
    parser.add_argument("--exhaust_test", action="store_true")
    args = get_combined_args(parser)         # train-time cfg_args of --model_path + command line (as render_s3.py)

    keep = {k: getattr(args, k) for k in ("model_path", "source_path", "loaded_pth")}   # render-time values that must survive the yaml
    # ---- from here: train.py's config handling, verbatim ----
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
    # ---- end of the copied block ----
    for k, v in keep.items():
        if getattr(args, k) != v:
            print("S3_V2_NOTE yaml set %s=%r; restored the render-time value %r" % (k, getattr(args, k), v))
            setattr(args, k, v)
    if not args.loaded_pth or not os.path.isfile(args.loaded_pth):
        print("S3_V2_FATAL need --loaded_pth <existing checkpoint>, got %r" % (args.loaded_pth,))
        sys.exit(2)
    safe_state(args.quiet)
    dataset, pipe = model.extract(args), pipeline.extract(args)

    time_duration = args.time_duration                                   # train.py line 39-41
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio, time_duration[1] / dataset.frame_ratio]
    print("S3_V2_CFG gaussian_dim=%s rot_4d=%s force_sh_3d=%s eval_shfs_4d=%s time_duration=%s frame_ratio=%s white_bg=%s num_pts=%s loaded_pth=%s" % (
          args.gaussian_dim, args.rot_4d, args.force_sh_3d, pipe.eval_shfs_4d, list(time_duration), dataset.frame_ratio, dataset.white_background, args.num_pts, args.loaded_pth))
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=time_duration, rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)   # train.py line 45
        scene = Scene(dataset, gaussians, shuffle=False, num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio, time_duration=time_duration)                                                          # train.py line 46 (+ shuffle=False)
        bg = torch.tensor([0.125, 0.216, 0.157] if args.render_bg == "green" else [1, 1, 1], dtype=torch.float32, device="cuda")
        os.makedirs(args.out_dir, exist_ok=True)
        tests = scene.test_cameras[1.0]                                  # list of Camera objects; no gt image is loaded
        rows, nskip = [], 0
        for idx in tqdm(range(len(tests)), desc="render"):
            cam = tests[idx]
            name = os.path.basename(cam.image_path).split('.')[0]        # camXX_NNNN
            outp = os.path.join(args.out_dir, name + ".png")
            if args.skip_existing and os.path.exists(outp):
                nskip += 1
                continue
            cam = cam.cuda()
            img = torch.clamp(render(cam, gaussians, pipe, bg)["render"], 0.0, 1.0)
            torchvision.utils.save_image(img, outp + ".tmp.png")
            os.replace(outp + ".tmp.png", outp)                           # atomic: a half-written png never counts as done
            rows.append("%s\t%.6f\t%d\t%d" % (name, float(getattr(cam, 'timestamp', float('nan'))), img.shape[2], img.shape[1]))
        open(os.path.join(args.out_dir, "render_index.tsv"), "a").write("\n".join(rows) + "\n")
        print("skipped_existing", nskip)
    print("RENDER_DONE", len(rows) + nskip)
