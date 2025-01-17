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


def msTPFP(line_pred, line_gt, threshold):
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(
        -1
    )  # are we permuting the lines to get
    test_diff = (line_pred[:, None, :, None] - line_gt[:, None]) ** 2

    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    choice = np.argmin(diff, 1)

    dist = np.min(diff, 1)

    hit = np.zeros(len(line_gt), bool)
    tp = np.zeros(len(line_pred), float)
    fp = np.zeros(len(line_pred), float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def ap(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)  # FIXME: why is there a maximum?

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])


GT_val = "evaluation/data/wireframe/valid/*.npz"
GT_train = "evaluation/data/wireframe/train/*_0_label.npz"


def line_score(path, threshold=5):
    preds = sorted(glob.glob(path))

    if path.split("/")[-2].split("_")[1] == "train":
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

        tp, fp = msTPFP(lcnn_line, gt_line, threshold)

        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(lcnn_score)

    lcnn_tp = np.concatenate(lcnn_tp)
    lcnn_fp = np.concatenate(lcnn_fp)
    lcnn_scores = np.concatenate(lcnn_scores)
    lcnn_index = np.argsort(-lcnn_scores)
    lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
    lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt

    return ap(lcnn_tp, lcnn_fp)


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
