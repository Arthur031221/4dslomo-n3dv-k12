# slomo_common.py -- shared constants / helpers of the slomo_full_k12 bundle (pure python, no torch).
import json, os

F0, NF, SPAN, L = 100, 60, 10.0, 33          # test window 100..159 ; official time_duration [0,10] ; 33 = test_lora.py num_frames
CHUNK_START = {'A': F0, 'B': F0 + NF - L}      # A = 100..132 ; B = 127..159 (overlap 127..132)

def grid_time(f):
    """time label of window frame f on the synchronous grid, same mapping as the stage-1 data (SPAN*(f-100)/60)."""
    return round(SPAN * (f - F0) / NF, 9)

def load_json(p):
    with open(p) as fh:
        return json.load(fh)

def split_by_cam(frames):
    out = {}
    for fr in frames:
        cam = os.path.basename(fr['file_path']).split('_')[0]
        out.setdefault(cam, []).append(fr)
    return out

def pick_gt(cam_frames, start):
    """real captured frame used as the 'GT frame 0' (CLIP + VAE first-frame condition) of a chunk starting at window frame `start`:
    the one whose TRUE capture time (slot + res) is closest to `start`; ties -> earlier."""
    return min(cam_frames, key=lambda fr: (abs(fr['frame'] + fr.get('res', 0.0) - start), fr['frame']))

def crop_geometry(w, h, side=1024):
    """exactly process_video.py crop_images(): resize short side to `side` (cv2 Lanczos, int() truncation), then centre crop side x side."""
    if w < h:
        new_w = side; new_h = int(h * (side / w))
    else:
        new_h = side; new_w = int(w * (side / h))
    return new_w / w, new_h / h, (new_w - side) // 2, (new_h - side) // 2, new_w, new_h

def stage2_header(head, side=1024):
    """intrinsics of the cropped 1024x1024 training images (corner-origin pixel convention, same as the stage-1 json)."""
    sx, sy, x0, y0, _, _ = crop_geometry(int(head['w']), int(head['h']), side)
    return dict(w=side, h=side, fl_x=head['fl_x'] * sx, fl_y=head['fl_y'] * sy, cx=head['cx'] * sx - x0, cy=head['cy'] * sy - y0)

def assignment(mode):
    """which diffusion output frame (chunk, idx) supplies the stage-2 training image of window frame f.
    returns list of (chunk, idx, frame, name_suffix)
      drop0 (default): drop output idx 0 (it re-draws the real first-frame image, whose capture time differs from the chunk start); A idx1..27 -> 101..127, B idx1..32 -> 128..159
      keep0          : demo-literal; A idx0..26 -> 100..126, B idx0..32 -> 127..159
      both           : every output frame of both chunks (overlap 127..132 twice, B copies get suffix 'b')"""
    out = []
    sA, sB = CHUNK_START['A'], CHUNK_START['B']
    if mode == 'drop0':
        out += [('A', i, sA + i, '') for i in range(1, 28)]
        out += [('B', i, sB + i, '') for i in range(1, L)]
    elif mode == 'keep0':
        out += [('A', i, sA + i, '') for i in range(0, 27)]
        out += [('B', i, sB + i, '') for i in range(0, L)]
    elif mode == 'both':
        out += [('A', i, sA + i, '') for i in range(0, L)]
        out += [('B', i, sB + i, 'b') for i in range(0, L)]
    else:
        raise SystemExit('bad assign mode ' + mode)
    return out
