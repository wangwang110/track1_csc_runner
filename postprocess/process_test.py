import re
import sys


def remove_sapce(sent):
    sent = re.sub("\s+(?![A-Za-z])", "", sent)
    return sent


def map2roformat(idx_li, src_li, trg_li):
    """
    :param srcs:
    :param trgs:
    :param ids:
    :return:
    """
    path = "./yaclc-csc-test.lbl"
    with open(path, "w", encoding="utf-8") as fw:
        for idx, src, trg in zip(idx_li, src_li, trg_li):
            res_li = []
            src_tokens = list(src)
            trg_tokens = list(trg)
            for i, item in enumerate(zip(src_tokens, trg_tokens)):
                s, t = item
                if s != t:
                    res_li.append(str(i + 1) + ", " + t)
            if len(res_li) == 0:
                fw.write(idx + ", 0" + "\n")
            else:
                fw.write(idx + ", " + ", ".join(res_li) + "\n")


def get_pairs_src(path, path_out_src):
    idx_li = []
    src_li = []
    with open(path, "r", encoding="utf-8") as f, open(path_out_src, "w", encoding="utf-8") as fw1:
        for line in f.readlines():
            idx, src = line.strip().split("\t")
            # src = remove_sapce(src)
            src_li.append(src)
            fw1.write(src + "\n")
            idx_li.append(idx)
    return idx_li, src_li


# 先把src的数据取出来
path = "../track1/track1_data/test/yaclc-csc_test.src"
path_out_src = "../track1/track1_data/test/yaclc-csc_test.txt"
idx_li, src_li = get_pairs_src(path, path_out_src)

# # 获得模型的结果
path_out_trg = sys.argv[1]
trg_li = [line.strip() for line in open(path_out_trg, "r", encoding="utf-8")]
map2roformat(idx_li, src_li, trg_li)
