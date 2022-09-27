import torch
import torch.nn as nn
from transformers import BertModel, BertForMaskedLM
from loss.focalloss import FocalLoss


class BERT_Model(nn.Module):

    def __init__(self, config, freeze_bert=False, tie_cls_weight=False):
        super(BERT_Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.label_ignore_id = config.label_ignore_id

        if tie_cls_weight:
            self.tie_cls_weight()

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            trg_ids=None,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = self.classifier(bert_output.last_hidden_state)
        loss = None
        if trg_ids is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=self.label_ignore_id)
            loss = loss_function(logits.view(-1, self.bert.config.vocab_size), trg_ids.view(-1))
        return loss, logits

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight
        # 共用还是只是初始化？
        # 这样是共用了吧

    def init_cls_weight(self):
        """
        一样吗
        :return:
        """
        word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        embeddings = nn.Parameter(word_embeddings_weight, True)
        self.classifier.weight = embeddings

        ##
        ## bert_embedding.word_embeddings = nn.Embedding(self.config.vocab_size, embedding_size, _weight=embeddings)
        ## 微调的时候不更新吗？？需要确认


class BERT_Mlm(nn.Module):

    def __init__(self, config, freeze_bert=False, tie_cls_weight=False):
        super(BERT_Mlm, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(config.pretrained_model)
        self.label_ignore_id = config.label_ignore_id

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            trg_ids=None,
    ):
        if trg_ids is not None:
            trg_ids[trg_ids == self.label_ignore_id] = -100

        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=trg_ids, return_dict=True, output_hidden_states=True)
        if trg_ids is None:
            outputs = (None, bert_outputs.logits)
        else:
            outputs = (bert_outputs.loss, bert_outputs.logits)

        return outputs


class BERT_Mlm_detect(nn.Module):

    def __init__(self, config, freeze_bert=False, tie_cls_weight=False):
        super(BERT_Mlm_detect, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(config.pretrained_model)
        self.label_ignore_id = config.label_ignore_id

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        hidden_size = self.bert.config.to_dict()['hidden_size']
        self.detection = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # 每个位置进行检测二分类

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            trg_ids=None,
            token_labels=None,
    ):
        if trg_ids is not None:
            trg_ids[trg_ids == self.label_ignore_id] = -100

        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=trg_ids, return_dict=True, output_hidden_states=True)
        if token_labels is not None:
            prob = self.detection(bert_outputs.hidden_states[-1])
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
            active_loss = attention_mask.view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = token_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())

        if trg_ids is None:
            outputs = (None, bert_outputs.logits)
        elif token_labels is not None:
            outputs = (bert_outputs.loss + det_loss, bert_outputs.logits)
        else:
            outputs = (bert_outputs.loss, bert_outputs.logits)

        return outputs
