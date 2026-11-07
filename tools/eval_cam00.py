#!/usr/bin/env python3
"""eval_cam00.py PRED_DIR GT_DATA_DIR OUT_TSV   (numpy + PIL)
PRED_DIR/frame_NNNN.png (NNNN = 0100..0159) vs GT_DATA_DIR/images/cam00_NNNN.png. Writes per-frame PSNR for the full 1352x1014 frame and for the central
square that the stage-2 training crop actually covers (columns [169,1183), all rows). Prints means. Quick sanity only; the real metrics are computed by us."""
import sys, numpy as np
from PIL import Image
pred, gt, outp = sys.argv[1:4]
rows = ['frame\tpsnr_full\tpsnr_centre']; A = []; Bc = []
def psnr(a, b): m = float(np.mean((a - b) ** 2)); return 99.0 if m == 0 else -10 * np.log10(m)
for f in range(100, 160):
    p = np.asarray(Image.open('%s/frame_%04d.png' % (pred, f)).convert('RGB'), dtype=np.float64) / 255
    g = np.asarray(Image.open('%s/images/cam00_%04d.png' % (gt, f)).convert('RGB'), dtype=np.float64) / 255
    assert p.shape == g.shape, (p.shape, g.shape)
    a = psnr(p, g); b = psnr(p[:, 169:1183], g[:, 169:1183]); A.append(a); Bc.append(b)
    rows.append('%d\t%.4f\t%.4f' % (f, a, b))
open(outp, 'w').write('\n'.join(rows) + '\n')
print('EVAL_CAM00 n=%d psnr_full=%.4f psnr_centre=%.4f' % (len(A), np.mean(A), np.mean(Bc)))
