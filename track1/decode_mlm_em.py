import torch
from transformers import BertTokenizer
from utils import init_dataloader,save_decode_result_lbl,save_decode_result_para
from model import BERT_Mlm
from tqdm import tqdm
import os
import argparse


class Decoder:
    def __init__(self, config):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.test_loader = init_dataloader(config.test_path, config, "test", self.tokenizer)
        self.model = BERT_Mlm(config)
        self.config = config

    def __forward_prop(self, dataloader, back_prop=True):
        collected_outputs = []
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items() if k!="token_labels"}
            _, logits = self.model(**batch)
            collected_outputs.append(logits.cpu())
        return collected_outputs

    def decode(self):
        model = self.model
        model_num = 0
        result = []
        for model_path in self.config.model_path.split(","):
            model_num += 1
            # model.load_state_dict(torch.load(model_path))

            model_dict = torch.load(model_path)
            # if 'classifier.weight' not in model_dict:
            #     model_dict['classifier.weight'] = model_dict['linear.weight']
            #     model_dict['classifier.bias'] = model_dict['linear.bias']
            used_model_dict = {k: model_dict[k] for k in model_dict if k in self.model.state_dict()}
            model.load_state_dict(used_model_dict)

            model.to(self.device)
            model.eval()
            with torch.no_grad():
                outputs = self.__forward_prop(dataloader=self.test_loader, back_prop=False)
                result.append(outputs)

        avg_ouputs = result[0]
        batch_num = len(result[0])
        collected_outputs = []
        for j in range(batch_num):
            for i in range(1, model_num):
                avg_ouputs[j] += result[i][j]
            avg_ouputs[j] = avg_ouputs[j] / model_num
            outputs = torch.argmax(avg_ouputs[j], dim=-1)
            for outputs_i in outputs:
                collected_outputs.append(outputs_i)

        save_decode_result_lbl(collected_outputs, self.test_loader.dataset.data, self.config.save_path)
        save_decode_result_para(collected_outputs, self.test_loader.dataset.data, self.config.save_path + ".pre")


def main(config):
    decoder = Decoder(config)
    decoder.decode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--label_ignore_id", default=0, type=int)

    args = parser.parse_args()
    main(args)
