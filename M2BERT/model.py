import math
import numpy as np
import random, time

import torch
import torch.nn as nn
from transformers import BertModel, LongformerModel, AutoModel

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        # print (x)
        return self.lut(x) * math.sqrt(self.d_model)


# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []      # [3,18,88,66]
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']        
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.long)
        
        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)


    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to bert 
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        return y
    
    def get_rand_tok(self):
        c1,c2,c3,c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array([random.choice(range(c1)),random.choice(range(c2)),random.choice(range(c3)),random.choice(range(c4))])


# class MidiLongformer(nn.Module):
#     def __init__(self, bertConfig, e2w, w2e):
#         super().__init__()
#         # print (bertConfig)
#         # self.longformer = LongformerModel(bertConfig)
#         self.longformer = AutoModel.from_config(bertConfig)
#         bertConfig.d_model = bertConfig.hidden_size
#         self.hidden_size = bertConfig.hidden_size
#         self.bertConfig = bertConfig

#         # print (self.longformer)

#         # token types: [Bar, Position, Pitch, Duration]
#         self.n_tokens = []      # [3,18,88,66]
#         # token types: [Bar, Position, TimeSig, Pitch, Duration]
#         # [3,18,5,88,66]
#         self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
#         for key in self.classes:
#             self.n_tokens.append(len(e2w[key]))
#         self.emb_sizes = [256, 256, 256, 256]
#         self.e2w = e2w
#         self.w2e = w2e

#         # for deciding whether the current input_ids is a <PAD> token
#         self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']        
#         self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.long)
#         self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.long)
        
#         # word_emb: embeddings to change token ids into embeddings
#         self.word_emb = []
#         for i, key in enumerate(self.classes):
#             self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
#         self.word_emb = nn.ModuleList(self.word_emb)

#         # linear layer to merge embeddings from different token types 
#         self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)


#     def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
#         # print (input_ids)
#         # convert input_ids into embeddings and merge them through linear layer
#         embs = []

#         for i, key in enumerate(self.classes):
#             embs.append(self.word_emb[i](input_ids[..., i]))
#         embs = torch.cat([*embs], dim=-1)
#         emb_linear = self.in_linear(embs)

#         # feed to bert 
#         # print (emb_linear.shape, attn_mask.shape)

#         y = self.longformer(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
#         #y = y.last_hidden_state         # (batch_size, seq_len, 768)
#         return y
    
#     def get_rand_tok(self):
#         # c1,c2,c3,c4,c5 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3], self.n_tokens[4]
#         c1,c2,c3,c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
#         return np.array([random.choice(range(c1)),random.choice(range(c2))
#             ,random.choice(range(c3)),random.choice(range(c4))])


def note2bar(y, bar_flag):
    # print (y.shape)
    longest_bar_counts = int(torch.max(torch.sum(bar_flag, dim=1)))
    # print (longest_bar_counts)
    bar_input = torch.zeros((y.shape[0], longest_bar_counts, y.shape[-1])).to(y.device)
    bar_attn_mask = torch.zeros((y.shape[0], longest_bar_counts)).to(y.device)

    for i in range(y.shape[0]):
        cur_bar_count = 0
        bar_list = torch.nonzero(bar_flag[i]).squeeze(1)
        # print (bar_list, bar_list.shape)
        for j in bar_list:
            bar_input[i][cur_bar_count] = y[i][j]
            bar_attn_mask[i][cur_bar_count] = 1
            cur_bar_count += 1

    return bar_input, bar_attn_mask

def bar2note(y, y_bar, bar_flag):
    for i in range(y.shape[0]):
        cur_bar_count = 0
        bar_list = torch.nonzero(bar_flag[i]).squeeze(1)
        # print (bar_list, bar_list.shape)
        for j in bar_list:
            y[i][j] = y_bar[i][cur_bar_count]
            cur_bar_count += 1
    return y


# class MidiModernBERT(nn.Module):
#     def __init__(self, bertConfig, e2w, w2e):
#         super().__init__()
#         # self.note_bert1 = AutoModel.from_config(bertConfig)
#         bertConfig.num_hidden_layers = bertConfig.num_hidden_layers // 3
#         self.note_bert1 = AutoModel.from_config(bertConfig)
#         self.bar_bert1 = AutoModel.from_config(bertConfig)
#         self.note_bert2 = AutoModel.from_config(bertConfig)

#         bertConfig.d_model = bertConfig.hidden_size
#         self.hidden_size = bertConfig.hidden_size
#         self.bertConfig = bertConfig

#         # print (self.longformer)

#         # token types: [Bar, Position, Pitch, Duration]
#         self.n_tokens = []      # [3,18,88,66]
#         # token types: [Bar, Position, TimeSig, Pitch, Duration]
#         # [3,18,5,88,66]
#         self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
#         for key in self.classes:
#             self.n_tokens.append(len(e2w[key]))
#         self.emb_sizes = [256, 256, 256, 256]
#         self.e2w = e2w
#         self.w2e = w2e

#         # for deciding whether the current input_ids is a <PAD> token
#         self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']        
#         self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.long)
#         self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.long)
        
#         # word_emb: embeddings to change token ids into embeddings
#         self.word_emb = []
#         for i, key in enumerate(self.classes):
#             self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
#         self.word_emb = nn.ModuleList(self.word_emb)

#         # linear layer to merge embeddings from different token types 
#         self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)


#     def forward(self, input_ids, attn_mask=None, bar_flag=None, output_hidden_states=True):
#         # print (input_ids)
#         # convert input_ids into embeddings and merge them through linear layer
#         embs = []
#         for i in range(len(self.classes)):
#             embs.append(self.word_emb[i](input_ids[:, :, i]))
#         embs = torch.cat(embs, dim=-1)
#         emb_linear = self.in_linear(embs)

#         if bar_flag is None:
#             bar_flag = (input_ids[:, :, 0] == 0).to(self.note_bert1.device)

#         y = self.note_bert1(inputs_embeds=emb_linear, attention_mask=attn_mask
#             , output_hidden_states=True)
#         y = y.hidden_states[-1]

#         bar_flag_cumsum = torch.cumsum(bar_flag, dim=1).long()
#         y = self.bar_bert1(inputs_embeds=y
#             , attention_mask=attn_mask, output_hidden_states=True
#             , position_ids=bar_flag_cumsum).hidden_states[-1]

#         y = self.note_bert2(inputs_embeds=y, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
#         return y

class MidiModernBERT(nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()
        # self.note_bert1 = AutoModel.from_config(bertConfig)
        self.note_bert1 = AutoModel.from_config(bertConfig)

        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # print (self.longformer)

        # token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []      # [3,18,88,66]
        # token types: [Bar, Position, TimeSig, Pitch, Duration]
        # [3,18,5,88,66]
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']        
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.long)
        
        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)


    def forward(self, input_ids, attn_mask=None, bar_flag=None, output_hidden_states=True):
        # print (input_ids)
        # convert input_ids into embeddings and merge them through linear layer
        embs = []
        for i in range(len(self.classes)):
            embs.append(self.word_emb[i](input_ids[:, :, i]))
        embs = torch.cat(embs, dim=-1)
        emb_linear = self.in_linear(embs)

        y = self.note_bert1(inputs_embeds=emb_linear, attention_mask=attn_mask
            , output_hidden_states=True)
        return y