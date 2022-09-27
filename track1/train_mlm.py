from utils import init_dataloader, save_decode_result_para, save_decode_result_lbl, csc_metrics, get_best_score
from transformers import BertTokenizer, AdamW, get_scheduler
from tqdm import tqdm
from model import BERT_Mlm, BERT_Mlm_detect
import torch
import os
import argparse
from random import seed


class Trainer:

    def __init__(self, config):
        self.config = config
        self.fix_seed(config.seed)
        print(config.__dict__)
        frozen_para = [
            "bert.embeddings.word_embeddings.weight",
            "bert.embeddings.position_embeddings.weight",
            "bert.embeddings.token_type_embeddings.weight",
            "bert.embeddings.LayerNorm.weight",
            "bert.embeddings.LayerNorm.bias",
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.0.attention.self.query.bias",
            "bert.encoder.layer.0.attention.self.key.weight",
            "bert.encoder.layer.0.attention.self.key.bias",
            "bert.encoder.layer.0.attention.self.value.weight",
            "bert.encoder.layer.0.attention.self.value.bias",
            "bert.encoder.layer.0.attention.output.dense.weight",
            "bert.encoder.layer.0.attention.output.dense.bias",
            "bert.encoder.layer.0.attention.output.LayerNorm.weight",
            "bert.encoder.layer.0.attention.output.LayerNorm.bias",
            "bert.encoder.layer.0.intermediate.dense.weight",
            "bert.encoder.layer.0.intermediate.dense.bias",
            "bert.encoder.layer.0.output.dense.weight",
            "bert.encoder.layer.0.output.dense.bias",
            "bert.encoder.layer.0.output.LayerNorm.weight",
            "bert.encoder.layer.0.output.LayerNorm.bias",
            "bert.encoder.layer.1.attention.self.query.weight",
            "bert.encoder.layer.1.attention.self.query.bias",
            "bert.encoder.layer.1.attention.self.key.weight",
            "bert.encoder.layer.1.attention.self.key.bias",
            "bert.encoder.layer.1.attention.self.value.weight",
            "bert.encoder.layer.1.attention.self.value.bias",
            "bert.encoder.layer.1.attention.output.dense.weight",
            "bert.encoder.layer.1.attention.output.dense.bias",
            "bert.encoder.layer.1.attention.output.LayerNorm.weight",
            "bert.encoder.layer.1.attention.output.LayerNorm.bias",
            "bert.encoder.layer.1.intermediate.dense.weight",
            "bert.encoder.layer.1.intermediate.dense.bias",
            "bert.encoder.layer.1.output.dense.weight",
            "bert.encoder.layer.1.output.dense.bias",
            "bert.encoder.layer.1.output.LayerNorm.weight",
            "bert.encoder.layer.1.output.LayerNorm.bias",
            "bert.encoder.layer.2.attention.self.query.weight",
            "bert.encoder.layer.2.attention.self.query.bias",
            "bert.encoder.layer.2.attention.self.key.weight",
            "bert.encoder.layer.2.attention.self.key.bias",
            "bert.encoder.layer.2.attention.self.value.weight",
            "bert.encoder.layer.2.attention.self.value.bias",
            "bert.encoder.layer.2.attention.output.dense.weight",
            "bert.encoder.layer.2.attention.output.dense.bias",
            "bert.encoder.layer.2.attention.output.LayerNorm.weight",
            "bert.encoder.layer.2.attention.output.LayerNorm.bias",
            "bert.encoder.layer.2.intermediate.dense.weight",
            "bert.encoder.layer.2.intermediate.dense.bias",
            "bert.encoder.layer.2.output.dense.weight",
            "bert.encoder.layer.2.output.dense.bias",
            "bert.encoder.layer.2.output.LayerNorm.weight",
            "bert.encoder.layer.2.output.LayerNorm.bias",
            "bert.encoder.layer.3.attention.self.query.weight",
            "bert.encoder.layer.3.attention.self.query.bias",
            "bert.encoder.layer.3.attention.self.key.weight",
            "bert.encoder.layer.3.attention.self.key.bias",
            "bert.encoder.layer.3.attention.self.value.weight",
            "bert.encoder.layer.3.attention.self.value.bias",
            "bert.encoder.layer.3.attention.output.dense.weight",
            "bert.encoder.layer.3.attention.output.dense.bias",
            "bert.encoder.layer.3.attention.output.LayerNorm.weight",
            "bert.encoder.layer.3.attention.output.LayerNorm.bias",
            "bert.encoder.layer.3.intermediate.dense.weight",
            "bert.encoder.layer.3.intermediate.dense.bias",
            "bert.encoder.layer.3.output.dense.weight",
            "bert.encoder.layer.3.output.dense.bias",
            "bert.encoder.layer.3.output.LayerNorm.weight",
            "bert.encoder.layer.3.output.LayerNorm.bias",
            "bert.encoder.layer.4.attention.self.query.weight",
            "bert.encoder.layer.4.attention.self.query.bias",
            "bert.encoder.layer.4.attention.self.key.weight",
            "bert.encoder.layer.4.attention.self.key.bias",
            "bert.encoder.layer.4.attention.self.value.weight",
            "bert.encoder.layer.4.attention.self.value.bias",
            "bert.encoder.layer.4.attention.output.dense.weight",
            "bert.encoder.layer.4.attention.output.dense.bias",
            "bert.encoder.layer.4.attention.output.LayerNorm.weight",
            "bert.encoder.layer.4.attention.output.LayerNorm.bias",
            "bert.encoder.layer.4.intermediate.dense.weight",
            "bert.encoder.layer.4.intermediate.dense.bias",
            "bert.encoder.layer.4.output.dense.weight",
            "bert.encoder.layer.4.output.dense.bias",
            "bert.encoder.layer.4.output.LayerNorm.weight",
            "bert.encoder.layer.4.output.LayerNorm.bias",
            "bert.encoder.layer.5.attention.self.query.weight",
            "bert.encoder.layer.5.attention.self.query.bias",
            "bert.encoder.layer.5.attention.self.key.weight",
            "bert.encoder.layer.5.attention.self.key.bias",
            "bert.encoder.layer.5.attention.self.value.weight",
            "bert.encoder.layer.5.attention.self.value.bias",
            "bert.encoder.layer.5.attention.output.dense.weight",
            "bert.encoder.layer.5.attention.output.dense.bias",
            "bert.encoder.layer.5.attention.output.LayerNorm.weight",
            "bert.encoder.layer.5.attention.output.LayerNorm.bias",
            "bert.encoder.layer.5.intermediate.dense.weight",
            "bert.encoder.layer.5.intermediate.dense.bias",
            "bert.encoder.layer.5.output.dense.weight",
            "bert.encoder.layer.5.output.dense.bias",
            "bert.encoder.layer.5.output.LayerNorm.weight",
            "bert.encoder.layer.5.output.LayerNorm.bias",
        ]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.train_dataloader = init_dataloader(config.train_path, config, "train", self.tokenizer)
        self.valid_dataloader = init_dataloader(config.dev_path, config, "dev", self.tokenizer)
        if config.use_cls_for_token == 1:
            self.model = BERT_Mlm_detect(config, config.freeze_bert, config.tie_cls_weight)
        else:
            self.model = BERT_Mlm(config, config.freeze_bert, config.tie_cls_weight)

        # self.model = BERT_Mlm(config, config.freeze_bert, config.tie_cls_weight)
        if config.frozen == 1:
            for name, p in self.model.named_parameters():
                if name[5:] in frozen_para:
                    print(p.requires_grad)
                    p.requires_grad = False

        # 加载模型继续训练
        if config.load_path != "":
            model_dict = torch.load(config.load_path)
            used_model_dict = {k: model_dict[k] for k in model_dict if k in self.model.state_dict()}
            self.model.load_state_dict(used_model_dict)

        self.model.to(self.device)
        # self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        if config.diff_lr == 1:
            # 实现分层学习率
            parameters_1 = [p for name, p in self.model.named_parameters() if name[5:] in frozen_para]
            parameters_2 = [p for name, p in self.model.named_parameters() if name[5:] not in frozen_para]
            self.optimizer = AdamW(
                [{"params": parameters_2}, {"params": parameters_1, "lr": config.lr}],
                lr=config.lr * 5
            )
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=config.lr)

        self.scheduler = self.set_scheduler()
        self.best_score = {"valid-c": 0, "valid-s": 0}
        self.best_epoch = {"valid-c": 0, "valid-s": 0}

    def fix_seed(self, seed_num):
        torch.manual_seed(seed_num)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed(seed_num)

    def set_scheduler(self):
        num_epochs = self.config.num_epochs
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        return lr_scheduler

    def __forward_prop(self, dataloader, back_prop=True):
        loss_sum = 0
        steps = 0
        collected_outputs = []
        print(len(dataloader))
        # for batch in tqdm(dataloader):
        for batch in dataloader:
            if self.config.use_cls_for_token == 1:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            else:
                batch = {k: v.to(self.device) for k, v in batch.items() if k!="token_labels"}
            loss, logits = self.model(**batch)
            outputs = torch.argmax(logits, dim=-1)
            for outputs_i in outputs:
                collected_outputs.append(outputs_i)
            loss_sum += loss.item()
            if back_prop:
                loss.backward()
                # print(loss.item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            steps += 1
        epoch_loss = loss_sum / steps
        return epoch_loss, collected_outputs

    def __save_ckpt(self, epoch):
        save_path = self.config.save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        path = os.path.join(save_path, self.config.tag + f"-epoch-{epoch}.pt")
        torch.save(self.model.state_dict(), path)

    def train(self):
        no_improve = 0
        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            train_loss, _ = self.__forward_prop(self.train_dataloader, back_prop=True)
            self.model.eval()
            with torch.no_grad():
                valid_loss, valid_output = self.__forward_prop(self.valid_dataloader, back_prop=False)
            print(f"train_loss: {train_loss}, valid_loss: {valid_loss}")
            if not os.path.exists(self.config.save_path + '/tmp/'):
                os.makedirs(self.config.save_path + '/tmp/')
            save_decode_result_para(valid_output, self.valid_dataloader.dataset.data,
                                    self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".txt")
            save_decode_result_lbl(valid_output, self.valid_dataloader.dataset.data,
                                   self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".lbl")
            try:
                char_metrics, sent_metrics = csc_metrics(
                    self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".lbl",
                    self.config.lbl_path)
                get_best_score(self.best_score, self.best_epoch, epoch,
                               char_metrics["Correction"]["F1"], sent_metrics["Correction"]["F1"])
                if self.config.save_best == 1:
                    # 预训练要多保存几个模型
                    if max(self.best_epoch.values()) == epoch:
                        self.__save_ckpt(epoch)
                else:
                    self.__save_ckpt(epoch)

            except:
                print("Decoded files cannot be evaluated.")

            print(f"curr epoch: {epoch} | curr best epoch {self.best_epoch}")
            print(f"best socre:{self.best_score}")
            print(f"no improve: {epoch - max(self.best_epoch.values())}")
            if (epoch - max(self.best_epoch.values())) >= self.config.patience:
                break


def main(config):
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", required=True, type=str)
    parser.add_argument("--use_cls_for_token", default=0, type=int)
    parser.add_argument("--train_path", required=True, type=str)
    parser.add_argument("--dev_path", required=True, type=str)
    parser.add_argument("--lbl_path", required=True, type=str)  ### 评测使用？
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--label_ignore_id", default=0, type=int)
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--patience", default=10, type=int)  ###
    parser.add_argument("--freeze_bert", default=False, type=bool)  ###
    parser.add_argument("--tie_cls_weight", default=False, type=bool)  ###
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--save_best', type=int, default=1)
    parser.add_argument('--frozen', type=int, default=0)
    parser.add_argument('--diff_lr', type=int, default=0)
    args = parser.parse_args()
    main(args)
