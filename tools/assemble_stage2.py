#!/usr/bin/env python3
"""assemble_stage2.py DATA_DIR DIFFUSION_DIR STAGE2_DIR [--assign drop0|keep0|both] [--add_real]
Builds the stage-2 4DSloMo dataset (what README step 3 does by hand with transforms_train_stage2.json):
  images/<cam>_<frame:04d>.png   symlinks to the diffusion outputs DIFFUSION_DIR/chunk{A,B}/out/images/<cam>_<idx:04d>.png (1024x1024)
  transforms_train.json          header = intrinsics of the 1024 centre crop (slomo_common.stage2_header), frames = diffusion images, pose = that camera's
                                 pose, time = grid_time(frame)  (the dance stage2 json likewise labels outputs on the virtual-frame grid, time = idx/36)
  transforms_test.json           copy of DATA_DIR's (cam00 held-out, original 1352x1014 intrinsics); images/cam00_*.png symlinked to the real GT
  points3d.ply                   copy of DATA_DIR's (README step 3 copies the stage-1 ply)
  --add_real  additionally writes the real captured frames, cropped exactly like process_video.py, as images/<cam>_<slot>r.png with their TRUE time (default: off,
              the official demo trains stage 2 on diffusion outputs only)"""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slomo_common import *

args = [a for a in sys.argv[1:] if not a.startswith('--')]
mode = sys.argv[sys.argv.index('--assign') + 1] if '--assign' in sys.argv else 'drop0'
if '--assign' in sys.argv: args.remove(mode)
add_real = '--add_real' in sys.argv
data, diff, s2 = [os.path.abspath(a) for a in args]
if os.path.exists(s2):
    raise SystemExit('EXISTS ' + s2)
tr = load_json(data + '/transforms_train.json')
head = {k: v for k, v in tr.items() if k != 'frames'}
h2 = stage2_header(head)
bycam = split_by_cam(tr['frames'])
os.makedirs(s2 + '/images')
frames = []
asg = assignment(mode)
for cam in sorted(bycam):
    pose = bycam[cam][0]['transform_matrix']
    for ch, idx, f, suf in asg:
        src = '%s/chunk%s/out/images/%s_%04d.png' % (diff, ch, cam, idx)
        assert os.path.exists(src), src
        name = '%s_%04d%s' % (cam, f, suf)
        os.symlink(src, '%s/images/%s.png' % (s2, name))
        frames.append(dict(file_path='images/' + name, transform_matrix=pose, time=grid_time(f), frame=f, chunk=ch, idx=idx))
    if add_real:
        import cv2
        for fr in bycam[cam]:
            img = cv2.imread(data + '/' + fr['file_path'] + '.png'); assert img is not None, fr['file_path']
            sx, sy, x0, y0, nw, nh = crop_geometry(img.shape[1], img.shape[0])
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)[y0:y0 + 1024, x0:x0 + 1024]
            name = '%s_%04dr' % (cam, fr['frame'])
            cv2.imwrite('%s/images/%s.png' % (s2, name), img)
            frames.append(dict(file_path='images/' + name, transform_matrix=pose, time=fr['time'], frame=fr['frame'], real=True))
json.dump(dict(h2, frames=frames), open(s2 + '/transforms_train.json', 'w'))
te = load_json(data + '/transforms_test.json')
for fr in te['frames']:
    n = os.path.basename(fr['file_path']); real = os.path.realpath('%s/%s.png' % (data, fr['file_path'])); assert os.path.exists(real), real
    os.symlink(real, '%s/images/%s.png' % (s2, n))
shutil.copy(data + '/transforms_test.json', s2 + '/transforms_test.json')
shutil.copy(data + '/points3d.ply', s2 + '/points3d.ply')
print('STAGE2_OK', s2, 'assign', mode, 'train frames', len(frames), 'test frames', len(te['frames']), 'header', h2)
