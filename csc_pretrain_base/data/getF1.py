#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


def sent_metric_correct(srcs, preds, targs):
    assert len(preds) == len(targs)
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for src_item, pred_item, targ_item in zip(srcs, preds, targs):
        assert len(pred_item) == len(targ_item)
        if src_item != targ_item:
            targ_p += 1
        if src_item != pred_item:
            pred_p += 1
            if pred_item == targ_item:
                tp += 1

        if pred_item == targ_item:
            hit += 1

    acc = hit / len(targs)
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    print(tp)
    print(pred_p)
    print(targ_p)
    print("acc:{}, p:{}, R:{}, f1:{}".format(round(acc, 3), round(p, 3), round(r, 3), round(f1, 3)))
    return acc, p, r, f1
