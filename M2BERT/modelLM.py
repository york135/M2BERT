import math
import numpy as np
import random

import torch
import torch.nn as nn
from transformers import BertModel


class MidiBertLM(nn.Module):
    def __init__(self, midibert, aux_output_dim):
        super().__init__()
        
        self.midibert = midibert
        self.mask_lm = MLM(self.midibert.e2w, self.midibert.n_tokens
            , self.midibert.hidden_size, aux_output_dim)

    def forward(self, x, attn_mask=None, bar_flag=None):
        x = self.midibert(x, attn_mask=attn_mask, bar_flag=bar_flag)
        return self.mask_lm(x)
    

class MLM(nn.Module):
    def __init__(self, e2w, n_tokens, hidden_size, aux_output_dim):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))

        # Add the next sentence prediction head
        # self.proj.append(nn.Linear(hidden_size, 16*12))
        self.proj.append(nn.Linear(hidden_size, aux_output_dim))

        self.proj = nn.ModuleList(self.proj)

        self.e2w = e2w
    
    def forward(self, y):
        # feed to bert 
        y = y.hidden_states[-1]
        
        # convert embeddings back to logits for prediction
        ys = []
        # for i, etype in enumerate(self.e2w):
        #     ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        for i in range(len(self.e2w) + 1):
            ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        return ys
    
