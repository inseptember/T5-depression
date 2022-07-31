import re
from typing import List

import numpy as np

START = '<extra_id_0>'
END = '<extra_id_1>'


class DataOut():
    def __init__(self, start_token=START, end_token=END, tokenizer=None):
        super(DataOut, self).__init__()
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.end_token = end_token

        to_remove_token_list = list()
        if tokenizer.bos_token:
            to_remove_token_list += [tokenizer.bos_token]
        if tokenizer.eos_token:
            to_remove_token_list += [tokenizer.eos_token]
        if tokenizer.pad_token:
            to_remove_token_list += [tokenizer.pad_token]
        self.to_remove_token_list = to_remove_token_list

    def batch_decode(self, tokens, convert=False):
        preds = np.where(tokens != -100, tokens, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        def clean_str(x_str):
            for to_remove_token in self.to_remove_token_list:
                x_str = x_str.replace(to_remove_token, '').strip()
                if convert:
                    x_str = self.clean_text(x_str)
            return x_str

        decoded_preds = [clean_str(x) for x in decoded_preds]
        return decoded_preds

    def clean_text(self, s):
        token_start = self.start_token
        token_end = self.end_token
        s = s.strip()
        if s.startswith(token_start + token_start):
            s = s[len(token_start):]
        if s.endswith(token_end + token_end):
            s = s[:-len(token_end)]
        pattern = '%s.*?%s' % (token_start, token_end)
        data = list(set(re.compile(pattern).findall(s)))
        data = [i.strip().replace(token_start, '').replace(token_end, '').split('<s/>')[0] if i.strip() != '' else ' ' for i in data]
        # data = [i.replace(token_start, '').replace(token_end, '') for i in data]
        return data

    def get_extract_metrics(self, pred_lns: List[str], tgt_lns: List[str], prefix=''):
        gold_list, pred_list = [], []
        for gold, pred in zip(tgt_lns, pred_lns):
            gold = self.clean_text(gold)
            pred = self.clean_text(pred)

            gold_list.append(gold)
            pred_list.append(pred)

        acc = 0
        gold_len = sum([len(i) for i in gold_list])
        test_len = sum([len(i) for i in pred_list])

        print(gold_len, test_len)
        for i, j in enumerate(pred_list):
            for k in j:
                if k in gold_list[i]:
                    acc += 1

        if acc == 0:
            return {prefix + 'P': 0,
                    prefix + 'R': 0,
                    prefix + 'F1': 0
                    }
        p = acc / test_len * 1.0
        r = acc / gold_len * 1.0
        return {prefix + 'P': p,
                prefix + 'R': r,
                prefix + 'F1': 2 * p * r / (p + r)
                }

