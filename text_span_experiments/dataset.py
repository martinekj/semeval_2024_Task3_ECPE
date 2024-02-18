import re
import numpy as np
from torch.utils.data import Dataset
import torch
import random
from transformers import BertTokenizerFast

text_spans_classes = [
    "whole_utterance", "first_part", "last_part", "middle_part", "other"
]

class TextSpansDatasetForTransformers(Dataset):
    """
   Parameters
   ----------
   texts: list of texts
   classes: list of labels
   tokenizerid: e.g. "bert-base-uncased"
   input_length: length of text input sequence
   """
    def __init__(self, texts, classes, tokenizerid, input_length):
        super().__init__()

        tokenizer = BertTokenizerFast.from_pretrained(tokenizerid)
        sep_text = tokenizer.sep_token  # '[SEP]'
        last_token = ""  # adds '[CLS] text [SEP]' by default

        x_input_ids = []
        x_attention_masks = []
        y = []

        for sentence, label in zip(texts, classes):
            input_txt = f"{sentence}{last_token}"
            x = tokenizer(input_txt, padding="max_length", truncation=True, max_length=input_length,
                          return_tensors="pt")
            x_input_ids.append(x.input_ids[0])
            x_attention_masks.append(x.attention_mask[0])

            label_index = text_spans_classes.index(label)
            y.append(torch.tensor(label_index, dtype=torch.long))

        self.max_length = input_length

        self.tokenizer_id = tokenizerid
        self.x_input_ids = x_input_ids
        self.x_attention_masks = x_attention_masks
        self.y = y
        self.length = len(self.y)
        self.num_classes = len(set(self.y))


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x_input_ids[idx]
        xm = self.x_attention_masks[idx]
        y = self.y[idx]
        return x, xm, y

    def get_input_len(self):
        return len(self.x_input_ids[0])
