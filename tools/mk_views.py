#!/usr/bin/env python3
"""mk_views.py DATA_DIR VIEWS_DIR   (stdlib only)
Builds a render-only "dataset" for render_views.py: training cameras (poses of DATA_DIR/transforms_train.json) x window frames 100..159 on the
synchronous grid (time = SPAN*(f-100)/60, i.e. residual-free), no ground truth. Layout:
  VIEWS_DIR/transforms_train.json   copy of DATA_DIR's (Scene needs a non-empty train set; never rendered)
  VIEWS_DIR/transforms_test.json    the view cameras  (file_path views/<cam>_<NNNN>)
  VIEWS_DIR/images -> DATA_DIR/images ; VIEWS_DIR/points3d.ply copy ; VIEWS_DIR/views/<cam>_<NNNN>.png -> dummy image (only its size is read)"""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slomo_common import *

data, views = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
if os.path.exists(views):
    raise SystemExit('EXISTS ' + views)
tr = load_json(data + '/transforms_train.json')
head = {k: v for k, v in tr.items() if k != 'frames'}
bycam = split_by_cam(tr['frames'])
os.makedirs(views + '/views')
os.symlink(data + '/images', views + '/images')
shutil.copy(data + '/transforms_train.json', views + '/transforms_train.json')
shutil.copy(data + '/points3d.ply', views + '/points3d.ply')
dummy = data + '/images/cam00_%04d.png' % F0
assert os.path.exists(dummy), dummy
frames = []
for cam in sorted(bycam):
    pose = bycam[cam][0]['transform_matrix']
    assert all(fr['transform_matrix'] == pose for fr in bycam[cam]), cam
    for f in range(F0, F0 + NF):
        name = '%s_%04d' % (cam, f)
        os.symlink(dummy, '%s/views/%s.png' % (views, name))
        frames.append(dict(file_path='views/' + name, transform_matrix=pose, time=grid_time(f), frame=f))
with open(views + '/transforms_test.json', 'w') as fh:
    json_dump = __import__('json').dump
    json_dump(dict(head, frames=frames), fh)
print('VIEWS_OK', views, 'cams', len(bycam), 'frames', len(frames))
