import sys

from torch.utils.data import DataLoader
from dataset import CSC_Dataset, Padding_in_batch
from eval_char_level import get_char_metrics
from eval_sent_level import get_sent_metrics
import string

vocab_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/vocab.txt"
vocab = []
with open(vocab_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    vocab.append(line.strip())


def init_dataloader(path, config, subset, tokenizer):
    sub_dataset = CSC_Dataset(path, config, subset)

    if subset == "train":
        is_shuffle = True
    else:
        is_shuffle = False

    collate_fn = Padding_in_batch(tokenizer.pad_token_id)

    data_loader = DataLoader(
        sub_dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        collate_fn=collate_fn
    )

    return data_loader


def csc_metrics(pred, gold):
    char_metrics = get_char_metrics(pred, gold)
    sent_metrics = get_sent_metrics(pred, gold)
    print("\n")
    return char_metrics, sent_metrics


def get_best_score(best_score, best_epoch, epoch, *params):
    for para, key in zip(params, best_score.keys()):
        if para > best_score[key]:
            best_score[key] = para
            best_epoch[key] = epoch
    return best_score, best_epoch


def save_decode_result_para(decode_pred, data, path):
    f = open(path, "w")
    results = []
    for i, (pred_i, src) in enumerate(zip(decode_pred, data)):
        src_i = src['input_ids']
        line = ""
        pred_i = pred_i[:len(src_i)]
        pred_i = pred_i[1:-1]
        src_i = src_i[1:-1]
        for id, ele in enumerate(pred_i):
            if vocab[ele] not in ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"] \
                    and vocab[ele] not in string.ascii_letters and len(vocab[ele]) == len(vocab[src_i[id]]):
                line += vocab[ele]
            else:
                line += src['src_text'][id]  # 使用bert分词，可以这么处理吗
        if 'trg_text' in src:
            f.write("ori:" + src['src_text'] + "\n")
            f.write("pre:" + line + "\n")
            f.write("trg:" + src['trg_text'] + "\n\n")
        else:
            f.write(line + "\n")

    f.close()


def save_decode_result_lbl(decode_pred, data, path):
    with open(path, "w") as fout:
        for pred_i, src in zip(decode_pred, data):
            src_i = src['input_ids']
            line = src['id'] + ", "
            pred_i = pred_i[:len(src_i)]
            no_error = True
            for id, ele in enumerate(pred_i):
                if ele != src_i[id]:
                    if vocab[ele] not in ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"] \
                            and vocab[ele] not in string.ascii_letters and len(vocab[ele]) == len(vocab[src_i[id]]):
                        no_error = False
                        line += (str(id) + ", " + vocab[ele] + ", ")
            if no_error:
                line += '0'
            line = line.strip(", ")
            fout.write(line + "\n")


if __name__ == "__main__":
    # test()
    pred = sys.argv[1]
    gold = sys.argv[2]
    csc_metrics(pred, gold)
