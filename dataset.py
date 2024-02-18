import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import config
from dialogues_data_loader import load_dialogues_json_data
from utils import utterances_to_dialogues

import utilz.config as cfg


AUDIO_VECS = None
VIDEO_VECS = None


def read_fvec_file(fpath):
    with open(fpath, "r") as f:
        lines = f.read().split("\n")

    while len(lines[-1]) == 0:
        lines.pop(-1)

    lines = [line.split(",") for line in lines]
    tofloat = lambda vs: [float(v) for v in vs]

    return {line[0]: tofloat(line[1:]) for line in lines}


def load_audio_video_vecs(audiofile="audio_vectors_nfcc2k.tsv", videofile="video_vectors.csv"):
    global AUDIO_VECS, VIDEO_VECS
    AUDIO_VECS = read_fvec_file(cfg.paths.data + "/" + audiofile)
    VIDEO_VECS = read_fvec_file(cfg.paths.data + "/" + videofile)


class DialogDataset(Dataset):
    def __init__(self, dataset_type, tokenizer="bert-base-cased", input_length=512):
        super().__init__()
        global AUDIO_VECS, VIDEO_VECS
        assert AUDIO_VECS is not None and VIDEO_VECS is not None

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if dataset_type is None:
            d = {}
        else:
            utts = load_dialogues_json_data(dataset_type=dataset_type)
            d = utterances_to_dialogues(utts)
            self.max_n_utt = max([len(v) for k, v in d.items()])

        self.input_length = input_length

        self.CLS = tokenizer.cls_token_id
        self.SEP = tokenizer.sep_token_id
        self.PAD = tokenizer.pad_token_id
        self.tokenizer = tokenizer

        assert config.emotion_categories[0] == '_NONE'
        assert config.speakers[0] == '_NONE'

        self.em2i = {}
        for i, v in enumerate(config.emotion_categories):
            self.em2i[v] = i

        self.sp2i = {}
        for i, v in enumerate(config.speakers):
            self.sp2i[v] = i

        dialogs = []
        for k, utterances in tqdm(d.items(), desc="loading dialogs"):
            dialogs.append(self.prepare_dialog(utterances))

        self.d = dialogs

    def __len__(self):
        return len(self.d)

    def _prepare_dialog(self, utterances):
        global AUDIO_VECS, VIDEO_VECS

        texts = []
        emotions = []
        speakers = []
        links = []
        audio = []
        video = []

        for utt in utterances:
            texts.append(utt.utterance_text)
            if utt.utterance_emotion not in self.em2i:
                emotions.append(0)
            else:
                emotions.append(self.em2i[utt.utterance_emotion])

            if utt.utterance_speaker not in self.sp2i:
                speakers.append(0)
            else:
                speakers.append(self.sp2i[utt.utterance_speaker])

            for link in utt.emotion_cause_links:
                srcid = int(link['source_id'].split("_")[1]) - 1
                trgid = int(link['target_id'].split("_")[1]) - 1
                # if srcid > trgid:
                #     print(f"Print backward link in utterance {utt.utterance_id}: {srcid} -> {trgid}")
                links.append([srcid, trgid])

            aud_vec = AUDIO_VECS[utt.utterance_id]
            vid_vec = VIDEO_VECS[utt.utterance_id]

            audio.append(aud_vec)
            video.append(vid_vec)

        tokens = self.tokenizer(texts, add_special_tokens=False)

        return (
            tokens['input_ids'],
            speakers,
            emotions,
            links,
            audio,
            video
        )

    def _pad(self, l, padval, padsize):
        assert len(l) <= padsize, f"input of len {len(l)} can not fit into {padsize} tokens given by input length"
        l = l[:padsize]
        l = l + [padval] * (padsize - len(l))
        return l

    def prepare_dialog(self, utterances, skip_links=False):
        assert len(utterances) <= self.max_n_utt, f"max_n_utt is set to {self.max_n_utt}, but {len(utterances)} occured"
        tokens, utt_speakers, utt_emotions, links, audio, video = self._prepare_dialog(utterances)

        dialog_tokens = [self.CLS]
        dialog_uttid = [0]

        for i, t in enumerate(tokens):
            t.append(self.SEP)
            dialog_tokens = dialog_tokens + t
            dialog_uttid = dialog_uttid + [i+1] * len(t)

        dialog_mask = [1] * len(dialog_tokens)
        utt_mask = [1] * (dialog_uttid[-1])

        dialog_tokens = self._pad(dialog_tokens, self.PAD, self.input_length)
        dialog_uttid = self._pad(dialog_uttid, 0, self.input_length)
        dialog_mask = self._pad(dialog_mask, 0, self.input_length)

        utt_mask = self._pad(utt_mask, 0, self.max_n_utt)
        utt_speakers = self._pad(utt_speakers, 0, self.max_n_utt)
        utt_emotions = self._pad(utt_emotions, 0, self.max_n_utt)
        utt_link_mat = np.zeros((self.max_n_utt, self.max_n_utt))
        if not skip_links:
            for l1, l2 in links:
                utt_link_mat[l1, l2] = 1

        utt_audio = self._pad(audio, [0.] * len(audio[0]), self.max_n_utt)
        utt_video = self._pad(video, [0.] * len(video[0]), self.max_n_utt)

        # all int64
        dialog_tokens = torch.tensor(dialog_tokens)     # tokens of the whole dialog: [CLS] utt1 [SEP] utt2 [SEP] ... uttN [SEP]
        dialog_uttid = torch.tensor(dialog_uttid)       # utterance index of the token, -1 stands for padding or 'belongs to no utterance'
        dialog_mask = torch.tensor(dialog_mask)         # mask for tokens

        utt_mask = torch.tensor(utt_mask)               # mask for the utterances of shape (self.max_n_utt)
        utt_speakers = torch.tensor(utt_speakers)       # speaker index of the utterance, 0 means no speaker
        utt_emotions = torch.tensor(utt_emotions)       # emmotion of the utterance
        utt_link_mat = torch.tensor(utt_link_mat, dtype=torch.float32)  # Adjacency matrix for the links in the dialog 'graph'  link is given by 1 at utt_link_mat[startutt][targetutt]

        utt_audio = torch.tensor(utt_audio, dtype=torch.float32)
        utt_video = torch.tensor(utt_video, dtype=torch.float32)

        return dialog_tokens, dialog_uttid, dialog_mask, utt_mask, utt_speakers, utt_audio, utt_video, utt_emotions, utt_link_mat

    def __getitem__(self, idx):
        return self.d[idx]


if __name__ == '__main__':
    ds = DialogDataset("test")

    for x in ds:
        print(x)

