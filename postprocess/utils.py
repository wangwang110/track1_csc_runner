import pickle
from char_smi import CharFuncs
from pypinyin import pinyin, lazy_pinyin, Style

char_smi = CharFuncs('char_meta.txt')


def pinyin_similar(src_pinyin, tgt_pinyin):
    for a in src_pinyin:
        for b in tgt_pinyin:
            if a + "g" == b or b + "g" == a:
                # 前后鼻音不分
                return True
            if a.replace("zh", "z").replace("sh", "s").replace("ch", "c") == b \
                    or b.replace("zh", "z").replace("sh", "s").replace("ch", "c") == a:
                # 平翘舌不分
                return True
            # 拼音相似增加 zhi，shi,chi
            if a in ["zhi", "chi", "shi"] and b in ["zhi", "chi", "shi"]:
                return True


def is_pinyin(src_char, tgt_char):
    p_threshold = 0.7
    try:
        p_sim = char_smi.pronunciation_similarity(src_char, tgt_char)
    except:
        print("不是汉字")
        print(src_char + "====" + tgt_char)
        return False

    src_pinyin = set(pinyin(src_char, style=Style.NORMAL, heteronym=True)[0])
    tgt_pinyin = set(pinyin(tgt_char, style=Style.NORMAL, heteronym=True)[0])

    if len(src_pinyin & tgt_pinyin) != 0 or p_sim >= p_threshold:
        return True
    if pinyin_similar(src_pinyin, tgt_pinyin):
        return True
    return False


def is_shape(src_char, tgt_char):
    s_threshold = 0.28
    try:
        v_sim = char_smi.shape_similarity(src_char, tgt_char)
    except:
        print("不在字形库")
        print(src_char + "====" + tgt_char)  # 不在字形库
        return None
    print(v_sim)
    if v_sim >= s_threshold:
        return True

    return False


base = "/home/track1_csc_runner/csc_pretrain_base/data_gen/ccl2022/"
confusion_set_pinyin = pickle.load(open(base + "/pinyin_confset.pkl", "rb"))
confusion_set_shape = pickle.load(open(base + "/other_confset.pkl", "rb"))
same_pinyin_words = pickle.load(
    open(base + "/chinese_homophone_word_simple.pkl", "rb"))
