#!/usr/bin/env python3
"""Evaluate sAP5, sAP10, sAP15 for LCNN
Usage:
    eval-sAP.py <path>...
    eval-sAP.py (-h | --help )

Examples:
    python eval-sAP.py logs/*/npz/000*

Arguments:
    <path>                           One or more directories from train.py

Options:
   -h --help                         Show this screen.
"""


import glob

import numpy as np
from docopt import docopt

# import lcnn.utils
import lcnn.metric
root_path = "/home/kallelis/PrimitiveExtraction/PrimitiveExtraction/"
GT_val = root_path + "Detection/LETR/data/synthetic_raw/labels/valid/*.npz"
GT_train = root_path + "Detection/LETR/data/synthetic_raw/labels/train/*_0_label.npz"

def line_score(path, threshold=5):
    preds = sorted(glob.glob(path))

    if path.split('/')[-2].split('_')[1] == 'train':
        print("######### Train #########")
        gts = sorted(glob.glob(GT_train))
        gts = gts[:501]
    else:
        gts = sorted(glob.glob(GT_val))



    n_gt = 0
    lcnn_tp, lcnn_fp, lcnn_scores = [], [], []
    for pred_name, gt_name in zip(preds, gts):
        with np.load(pred_name) as fpred:
            lcnn_line = fpred["lines"][:, :, :2]
            lcnn_score = fpred["score"]
        with np.load(gt_name) as fgt:
            try:
                gt_line = fgt["lpos"][:, :, :2]
            except IndexError:
                print("########### image has no lines")
                print(fgt["lpos"])
        n_gt += len(gt_line)

        for i in range(len(lcnn_line)):
            if i > 0 and (lcnn_line[i] == lcnn_line[0]).all():
                lcnn_line = lcnn_line[:i]
                lcnn_score = lcnn_score[:i]
                break
        print("##################")
        print(len(lcnn_line))
        print(lcnn_line.shape)
        print("##################")
        # exit()
        tp, fp = lcnn.metric.msTPFP(lcnn_line, gt_line, threshold)
        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(lcnn_score)

    lcnn_tp = np.concatenate(lcnn_tp)
    lcnn_fp = np.concatenate(lcnn_fp)
    lcnn_scores = np.concatenate(lcnn_scores)
    lcnn_index = np.argsort(-lcnn_scores)
    lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
    lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt

    return lcnn.metric.ap(lcnn_tp, lcnn_fp)


if __name__ == "__main__":
    args = docopt(__doc__)

    def work(path):
        print(f"Working on {path}")
        return [100 * line_score(f"{path}/*.npz", t) for t in [5, 10, 15]]

    dirs = sorted(sum([glob.glob(p) for p in args["<path>"]], []))
    results = [work(dirs[0])]    

    # results = lcnn.utils.parmap(work, dirs)

    for d, msAP in zip(dirs, results):
        print(f"{d}: {msAP[0]:2.1f} {msAP[1]:2.1f} {msAP[2]:2.1f}")
