#!/usr/bin/env python3
"""prep_chunks.py DATA_DIR RENDERS_DIR CHUNK_DIR A|B   (stdlib only)
Builds the input folder that the OFFICIAL process_video.py / test_lora.py expect (README l.63-67 layout):
  CHUNK_DIR/in/test/ours_None/renders/<cam>_<idx:03d>.png   idx = 0..32  -> RENDERS_DIR/<cam>_<frame:04d>.png   (symlinks; frame = start+idx)
  CHUNK_DIR/in/test/ours_None/gt/<cam>_0000.png             the real captured frame nearest in true time to the chunk start (symlink)
  CHUNK_DIR/cams.txt , CHUNK_DIR/plan.json (which real frame was used per camera, its time offset to the chunk start)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slomo_common import *

data, rend, cdir, ch = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]), sys.argv[4]
start = CHUNK_START[ch]
if os.path.exists(cdir + '/in'):
    raise SystemExit('EXISTS ' + cdir + '/in')
bycam = split_by_cam(load_json(data + '/transforms_train.json')['frames'])
R = cdir + '/in/test/ours_None/renders'; G = cdir + '/in/test/ours_None/gt'
os.makedirs(R); os.makedirs(G)
plan = {}
for cam in sorted(bycam):
    for i in range(L):
        src = '%s/%s_%04d.png' % (rend, cam, start + i)
        assert os.path.exists(src), src
        os.symlink(src, '%s/%s_%03d.png' % (R, cam, i))
    g = pick_gt(bycam[cam], start)
    src = data + '/' + g['file_path'] + '.png'
    assert os.path.exists(src), src
    os.symlink(src, '%s/%s_0000.png' % (G, cam))
    plan[cam] = dict(gt_file=g['file_path'], gt_slot=g['frame'], gt_true_time_frames=round(g['frame'] + g.get('res', 0.0), 4),
                     offset_to_chunk_start_frames=round(g['frame'] + g.get('res', 0.0) - start, 4))
open(cdir + '/cams.txt', 'w').write('\n'.join(sorted(bycam)) + '\n')
json.dump(dict(chunk=ch, start=start, frames=[start, start + L - 1], plan=plan), open(cdir + '/plan.json', 'w'), indent=1)
print('CHUNK_OK', ch, start, 'cams', len(bycam), 'max |gt offset| (frames) =', max(abs(v['offset_to_chunk_start_frames']) for v in plan.values()))
