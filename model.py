import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

import config
from dataset import DialogDataset, load_audio_video_vecs


class LinkAttModule(nn.Module):     # lightweight BertSelfAttention
    def __init__(self, num_attention_heads, hidden_size, sqrtatt, aggregation):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)

        self.sqrtatt = sqrtatt
        if aggregation == 'mean':
            self.aggfunc = torch.mean
        elif aggregation == 'max':
            self.aggfunc = lambda x, dim: torch.max(x, dim=dim)[0]
        elif aggregation == 'min':
            self.aggfunc = lambda x, dim: torch.min(x, dim=dim)[0]
        else:
            raise NotImplementedError

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.sqrtatt:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = self.aggfunc(attention_scores, dim=1)  # avg accross multiple heads

        # Apply the attention mask
        attention_scores = attention_scores * attention_mask.unsqueeze(dim=1)   # mask right part
        attention_scores = attention_scores * attention_mask.unsqueeze(dim=-1)  # mask bottom part
        # attention_scores = torch.triu(attention_scores) # taking only upper triangular matrix (similar to masked attention) - edit backward links are possible so no triu
        return attention_scores


class FusionModule(nn.Module):
    def __init__(self, layers, hidden_size, repr_dim, audio, video, audio_dim=2048, video_dim=2048, aggreg="cls", proj_lrelu=False):
        super().__init__()

        self.audio = audio
        self.video = video
        self.aggreg = aggreg
        self.proj_lrelu = proj_lrelu

        self.config = BertConfig(
            vocab_size=0,
            hidden_size=hidden_size,
            num_hidden_layers=layers,
            num_attention_heads=hidden_size//64,
            intermediate_size=hidden_size*4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=4,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None
        )
        self.fuser = BertEncoder(self.config)

        self.repr_proj = torch.nn.Linear(repr_dim, hidden_size)
        if self.audio:
            self.audio_proj = torch.nn.Linear(audio_dim, hidden_size)
        if self.video:
            self.video_proj = torch.nn.Linear(video_dim, hidden_size)

        if self.aggreg == 'cls':
            self.clsprompt = torch.nn.Parameter(torch.empty((1, hidden_size)))  # learnable cls token
            nn.init.xavier_uniform_(self.clsprompt)
        elif self.aggreg == 'spkrcls':
            nspeakers = len(config.speakers)
            self.speaker_embeddings = nn.Embedding(nspeakers, hidden_size)
        elif self.aggreg == 'mean':
            self.aggfunc = torch.mean
        elif self.aggreg == 'max':
            self.aggfunc = lambda x, dim: torch.max(x, dim=dim)[0]
        elif self.aggreg == 'min':
            self.aggfunc = lambda x, dim: torch.min(x, dim=dim)[0]
        else:
            raise NotImplementedError

        if self.proj_lrelu:
            self.proj_act_f = torch.nn.LeakyReLU()

    def forward(self, utt_repr, utt_audio, utt_video, utt_speakers):
        ftrs = [self.repr_proj(utt_repr)]
        if self.audio:
            ftrs.append(self.audio_proj(utt_audio))
        if self.video:
            ftrs.append(self.video_proj(utt_video))

        if self.proj_lrelu:
            ftrs = [self.proj_act_f(ftr) for ftr in ftrs]

        batch, n_utts, dim = ftrs[0].shape
        concatshp = (batch * n_utts, 1, dim)
        ftrs = [torch.reshape(ftr, concatshp) for ftr in ftrs]
        if self.aggreg == 'cls':
            clsprompt = self.clsprompt.expand(batch * n_utts, 1, dim)
            ftrs = [clsprompt] + ftrs
        elif self.aggreg == 'spkrcls':
            spkrprompt = self.speaker_embeddings(utt_speakers)
            ftrs = [torch.reshape(spkrprompt, concatshp)] + ftrs

        fusioninput = torch.concatenate(ftrs, dim=1)

        # shape (x, 3, dim)
        fused_repr = self.fuser(
            hidden_states=fusioninput
        ).last_hidden_state

        if self.aggreg in['cls', 'spkrcls']:
            res = fused_repr[:, 0]
        else:
            res = self.aggfunc(fused_repr, 1)

        res = torch.reshape(res, (batch, n_utts, dim))
        return res


class DialogModel(nn.Module):
    def __init__(self, model="bert-base-cased", speaker_embeds=True, utterance_embeds=True, utterance_position_embeds=True, enc_hidden_layers=6, max_utts=34, sqrtatt=True, aggregation='mean', fusion_layers=6, fusion_size=768,
                 audio=True, video=True, fusion_aggreg="cls", proj_lrelu=False):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_utts = max_utts
        self.backbone = AutoModel.from_pretrained(model)

        nclasses = len(config.emotion_categories)
        nspeakers = len(config.speakers)
        hiddensize = self.backbone.config.hidden_size

        self.utterance_embeds = utterance_embeds
        if self.utterance_embeds:
            self.utterance_embeddings = nn.Embedding(max_utts+1, hiddensize, padding_idx=0)
            nn.init.constant_(self.utterance_embeddings.weight, 0.0)  # zero utterance embeddings at start

        if audio or video:
            self.fusion_module = FusionModule(layers=fusion_layers, hidden_size=fusion_size, repr_dim=hiddensize, audio=audio, video=video, aggreg=fusion_aggreg, proj_lrelu=proj_lrelu)
            configdict = self.fusion_module.config.to_dict()
        else:   # ignore fusion since there is no multimodality
            self.fusion_module = lambda a, b, c, d: a
            fusion_size = hiddensize
            configdict = self.backbone.config.to_dict()

        self.speaker_embeds = speaker_embeds
        if self.speaker_embeds:
            self.speaker_embeddings = nn.Embedding(nspeakers, fusion_size, padding_idx=0)
            nn.init.constant_(self.speaker_embeddings.weight, 0.0)  # zero speaker embeddings at start

        self.utt_classifier = nn.Sequential(  # learnable classifier head for emotions
            nn.Linear(fusion_size, fusion_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fusion_size, nclasses)
        )

        self.utterance_position_embeds = utterance_position_embeds
        if self.utterance_position_embeds:
            self.utterance_position_embeddings = nn.Embedding(max_utts, fusion_size, padding_idx=None)
            nn.init.constant_(self.utterance_position_embeddings.weight, 0.0)
            self.pos_ids = torch.arange(max_utts).expand((1, -1)).to(self.device)

        self.enc_hidden_layers = enc_hidden_layers
        if self.enc_hidden_layers > 0:
            configdict["num_hidden_layers"] = enc_hidden_layers
            configdict["max_position_embeddings"] = max_utts
            self.utt_encoder = BertEncoder(BertConfig(**configdict))     # to encode utterances

        self.utt_attention = LinkAttModule(configdict["num_attention_heads"], fusion_size, sqrtatt=sqrtatt, aggregation=aggregation)  # self attention matrix as adjacency matrix

        self.to(self.device)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def save_learnable_params(self, path):
        sd = self.state_dict()
        rmkeys = [name for name, param in self.named_parameters() if not param.requires_grad]
        for k in rmkeys:
            sd.pop(k)
        torch.save(sd, path)

    def load_learnable_params(self, path):
        mk = self.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        for k in mk.missing_keys:
            assert k.startswith("backbone.")

    def forward(self, dialog_tokens, dialog_uttid, dialog_mask, utt_mask, utt_speakers, utt_audio, utt_video, *args):
        """
        dialog_tokens (torch.LongTensor of shape (batch_size, sequence_length)) — Indices of input sequence tokens in the vocabulary.
        dialog_mask (torch.FloatTensor of shape (batch_size, sequence_length)) — Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        1 for tokens that are not masked,
        0 for tokens that are masked.
        """
        #   preparation of input_embeds (batch_size, sequence_length, hidden_size)
        embeds = self.backbone.get_input_embeddings()(
            dialog_tokens.to(self.device)
        )  # pretrained input embeddings of frozen model

        dialog_uttid = dialog_uttid.to(self.device)
        if self.utterance_embeds:
            utt_embeds = self.utterance_embeddings(dialog_uttid)
            embeds += utt_embeds

        # forward with backbone
        x = self.backbone(inputs_embeds=embeds, attention_mask=dialog_mask.to(self.device)).last_hidden_state

        # AVG tokens according to dialog_uttid
        n_utts = utt_mask.shape[-1]
        dialog_uttid = dialog_uttid.unsqueeze(-1)
        utt_avg_representations = []
        for i in range(1, n_utts + 1):
            cond = (dialog_uttid == i)
            tmp = (x * cond).sum(dim=1)    # sum tokens of the given utterance
            tmp = tmp / (cond.sum(axis=1) + self.eps)   # make avg
            utt_avg_representations.append(tmp.unsqueeze(1))

        utt_avg_representations = torch.concatenate(utt_avg_representations, dim=1)     # utt avg representation of shape (batch, n_utts, hidden_size)

        utt_speakers = utt_speakers.to(self.device)
        utt_avg_representations = self.fusion_module(utt_avg_representations, utt_audio.to(self.device), utt_video.to(self.device), utt_speakers)

        if self.speaker_embeds:     # add speaker embeddings to representations
            speaker_embeds = self.speaker_embeddings(utt_speakers)
            utt_avg_representations += speaker_embeds

        utt_mask = utt_mask.to(self.device)

        utt_classes_logits = self.utt_classifier(utt_avg_representations)
        utt_classes_logits = utt_classes_logits * utt_mask.unsqueeze(dim=-1)    # mask the utterances to prevent gradient backpropagation

        if self.utterance_position_embeds:
            utt_pos_embeds = self.utterance_position_embeddings(self.pos_ids[:, :n_utts])
            utt_avg_representations = torch.add(utt_avg_representations, utt_pos_embeds)
            # utt_avg_representations += utt_pos_embeds   # can not be inplace

        if self.enc_hidden_layers > 0:
            # now using the transformer on AVG utt
            dtype = utt_avg_representations.dtype
            extended_attention_mask = utt_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
            utt_avg_representations = self.utt_encoder(
                hidden_states=utt_avg_representations,
                attention_mask=extended_attention_mask,
            ).last_hidden_state

        utt_links_logits = self.utt_attention(
            hidden_states=utt_avg_representations,
            attention_mask=utt_mask
        )

        return utt_classes_logits, utt_links_logits




if __name__ == '__main__':
    config.speakers = config.main_speakers

    load_audio_video_vecs()

    model = DialogModel()

    dataset = DialogDataset("dev")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    classes_loss = nn.CrossEntropyLoss()
    links_loss = nn.BCEWithLogitsLoss()

    for x in dataloader:
        utt_classes_gt = x[-2].to(model.device)
        utt_links_gt = x[-1].to(model.device)
        utt_classes_logits, utt_links_logits = model(*x)

        closs = classes_loss(utt_classes_logits.reshape((-1, utt_classes_logits.shape[-1])), utt_classes_gt.flatten())
        lloss = links_loss(utt_links_logits, utt_links_gt)

        loss = closs + lloss
        print(loss)




