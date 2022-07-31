from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = CNN({
                    'cnn_out_channels' : 512,
                    'cnn_filter_sizes' : [3, 9, 27],
                    'cnn_hidden_dim1' : 1024,
                    'cnn_conv_stride' : 2,
                    'cnn_pool_stride' : 8,
                    'cnn_dropout' : [False, False],
                    'classes': config.num_labels
                })

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Convlution，MaxPooling
def out_size(sequence_length, filter_size, padding=0, dilation=1, stride=1):
    length = sequence_length + 2 * padding - dilation * (filter_size - 1) - 1
    length = int(length / stride)
    return length + 1


class CNN(torch.nn.Module):

    def __init__(self, params, gat=None):
        super(CNN, self).__init__()

        # self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        poolingLayer_out_size = 0

        self.dropout = params['cnn_dropout']
        self.filter_size = params['cnn_filter_sizes']

        if bool(self.dropout[0]):
            self.drp1 = nn.Dropout(p=self.dropout[0])
        if bool(self.dropout[1]):
            self.drp2 = nn.Dropout(p=self.dropout[1])

        for fsz in self.filter_size:
            l_conv = nn.Conv1d(params['embedding_dim'], params['cnn_out_channels'], fsz,
                               stride=params['cnn_conv_stride'])
            torch.nn.init.xavier_uniform_(l_conv.weight)

            l_pool = nn.MaxPool1d(params['cnn_pool_stride'], stride=params['cnn_pool_stride'])
            l_out_size = out_size(params['sequence_length'], fsz, stride=params['cnn_conv_stride'])
            pool_out_size = int(l_out_size * params['cnn_out_channels'] / params['cnn_pool_stride'])
            poolingLayer_out_size += pool_out_size

            self.conv_layers.append(l_conv)
            self.pool_layers.append(l_pool)

        self.linear1 = nn.Linear(poolingLayer_out_size, params['cnn_hidden_dim1'])
        self.linear2 = nn.Linear(params['cnn_hidden_dim1'], params['classes'])
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, texts):

        # texts = self.bert(texts)[0].detach_()

        texts = texts.permute(0, 2, 1)

        if bool(self.dropout[0]):
            texts = self.drp1(texts)

        conv_out = []

        for i in range(len(self.filter_size)):
            outputs = self.conv_layers[i](texts)
            outputs = outputs.view(outputs.shape[0], 1, outputs.shape[1] * outputs.shape[2])
            outputs = self.pool_layers[i](outputs)
            outputs = nn.functional.relu(outputs)
            outputs = outputs.view(outputs.shape[0], -1)
            conv_out.append(outputs)
            del outputs

        if len(self.filter_size) > 1:
            outputs = torch.cat(conv_out, 1)
        else:
            outputs = conv_out[0]

        outputs = self.linear1(outputs)

        outputs = nn.functional.relu(outputs)

        if bool(self.dropout[1]):
            outputs = self.drp2(outputs)

        outputs = self.linear2(outputs)

        return outputs
