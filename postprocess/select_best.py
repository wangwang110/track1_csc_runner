# coding: utf-8

"""
@File    : analysis_test.py.py
@Time    : 2022/7/18 16:16
@Author  : liuwangwang
@Software: PyCharm
"""
import sys
from csc_match import CSCmatch

obj = CSCmatch()


def use_lm_select(path1, path2, path_out):
    pre1 = [line.strip().replace(" ", "") for line in open(path1).readlines()]
    pre2 = [line.strip().replace(" ", "") for line in open(path2).readlines()]
    with open(path_out, "w", encoding="utf-8") as fw:
        for p1, p2 in zip(pre1, pre2):
            p1_str = " ".join(list(p1))
            p2_str = " ".join(list(p2))
            scores = obj.getscores([p1_str, p2_str])
            if scores[1] > scores[0] + 0.5:
                print(p1)
                print(p2)
                print(scores)
                fw.write(p2 + "\n")
            else:
                fw.write(p1 + "\n")


path1 = sys.argv[1]
path2 = sys.argv[2]
path_out = sys.argv[3]
use_lm_select(path1, path2, path_out)
