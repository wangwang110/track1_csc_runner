# coding: utf-8

import sys


def map2test(path, path_out):
    path_src = "../track1/track1_data/test/yaclc-csc_test.src"
    with open(path_src, "r", encoding="utf-8") as f, open(path, "r", encoding="utf-8") as fr, \
            open(path_out, "w", encoding="utf-8") as fw:
        id_li = []
        for line in f.readlines():
            idx, sent = line.strip().split("\t")
            id_li.append(idx)

        i = 0
        for line in fr.readlines():
            src = line.strip()
            fw.writelines(id_li[i] + "\t" + src + "\n")
            i += 1


path = sys.argv[1]
path_out = sys.argv[2]
map2test(path, path_out)
