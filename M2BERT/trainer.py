import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
from pytorch_optimizer import StableAdamW
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random
import tqdm, time
import sys
import shutil
import copy

from M2BERT.modelLM import MidiBertLM
from collections import OrderedDict

class BERTTrainer:
    def __init__(self, midibert, train_dataloader, valid_dataloader, 
                lr, batch, max_seq_len, mask_percent, cpu, cuda_devices=None, aux_output_dim=1):
        # print (max_seq_len)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        self.midibert = midibert        # save this for ckpt
        self.model = MidiBertLM(midibert, aux_output_dim=aux_output_dim)

        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('# total parameters:', self.total_params)

        if torch.cuda.device_count() > 1 and not cpu:
            print (cuda_devices)
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        
        self.model = self.model.to(self.device)

        self.optim = StableAdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.batch = batch
        self.max_seq_len = max_seq_len
        self.mask_percent = mask_percent
        self.Lseq = [i for i in range(self.max_seq_len)]
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.chroma_loss_func = nn.MSELoss(reduction='none')
        self.next_sentence_loss_func = nn.BCEWithLogitsLoss()
    
    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / max(torch.sum(loss_mask), 0.1)

        return loss

    def compute_chroma_loss(self, predict, target, loss_mask):
        loss = torch.sum(self.chroma_loss_func(predict, target), dim=2)
        loss = loss * loss_mask
        loss = torch.sum(loss) / max(torch.sum(loss_mask), 0.1)
        return loss

    def get_mask_ind(self, cur_max_notes):
        mask_ind = random.sample(self.Lseq[:cur_max_notes]
                , round(float(cur_max_notes) * self.mask_percent))

        # random.shuffle(mask_ind)
        # mask80 = mask_ind[:int(round(len(mask_ind)*0.5))]
        # rand10 = mask_ind[int(round(len(mask_ind)*0.5)):int(round(len(mask_ind)*0.9))]
        # cur10 = mask_ind[int(round(len(mask_ind)*0.9)):]
        # return mask80, rand10, cur10
        return [], mask_ind, []

    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, self.max_seq_len)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = self.iteration(self.valid_data, self.max_seq_len, train=False)
        return valid_loss, valid_acc

    def iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_losses = [0]*(len(self.midibert.e2w)+1), 0
        total_chroma_losses = 0
        
        for batch_id, ori_seq_batch in enumerate(pbar):
            batch = ori_seq_batch[0].shape[0]
            ori_seq_batch_data = ori_seq_batch[0]  # (batch, seq_len, 4) 
            bar_chroma_gt = ori_seq_batch[1].to(self.device)

            input_ids = copy.deepcopy(ori_seq_batch_data)
            loss_mask = torch.zeros(batch, max_seq_len)

            loss_chroma_mask = (input_ids[:, :, 0] != self.midibert.bar_pad_word).to(self.device)
            ori_seq_batch_data = ori_seq_batch_data.to(self.device)

            # 0: bar_start; 1: not bar_start; 2 and 3: padding
            bar_flag = (input_ids[:, :, 0] == 0)
            # print ('2', time.time())
            max_notes = torch.sum((input_ids[:, :, 0] != self.midibert.bar_pad_word), dim=1)

            c2_min = torch.clip(input_ids[:,:,1] - 4, min=0)
            c2_max = torch.clip(input_ids[:,:,1] + 4, max=self.midibert.n_tokens[1] - 2)

            c3_min = torch.clip(input_ids[:,:,2] - 12, min=0)
            c3_max = torch.clip(input_ids[:,:,2] + 12, max=self.midibert.n_tokens[2] - 2)

            c4_min = torch.clip(input_ids[:,:,3] - 12, min=0)
            c4_max = torch.clip(input_ids[:,:,3] + 12, max=self.midibert.n_tokens[3] - 2)

            for b in range(batch):
                # get index for masking
                mask80, rand10, cur10 = self.get_mask_ind(max_notes[b])
                # apply mask, random, remain current token
                mask_word = torch.tensor(self.midibert.mask_word_np)
                for i in mask80:
                    input_ids[b][i][1:] = mask_word[1:]
                    loss_mask[b][i] = 1
                for i in rand10:
                    rand_word = torch.tensor(np.array([random.choice(range(0, self.midibert.n_tokens[0] - 2))
                        , random.choice(range(c2_min[b][i], c2_max[b][i]))
                        , random.choice(range(c3_min[b][i], c3_max[b][i]))
                        , random.choice(range(c4_min[b][i], c4_max[b][i]))]))
                    input_ids[b][i][1:] = rand_word[1:]
                    loss_mask[b][i] = 1
                for i in cur10:
                    loss_mask[b][i] = 1 

            loss_mask = loss_mask.to(self.device)
            # avoid attend to pad word
            attn_mask = (input_ids[:, :, 0] != self.midibert.bar_pad_word).float()   # (batch, seq_len)

            if torch.cuda.is_available():
                input_ids = input_ids.to(self.device)
                bar_flag = bar_flag.to(self.device)
                attn_mask = attn_mask.to(self.device)
            
            y = self.model(input_ids, attn_mask=attn_mask, bar_flag=bar_flag)

            # get the most likely choice with max
            outputs = []
            for i, etype in enumerate(self.midibert.e2w):
                output = torch.argmax(y[i], dim=-1, keepdim=True).detach()
                outputs.append(output)
            outputs = torch.cat(outputs, dim=-1)

            # accuracy
            all_acc = []
            for i in range(4):
                acc = torch.sum((ori_seq_batch_data[:,:,i] == outputs[:,:,i]).float() * loss_mask)
                acc /= max(torch.sum(loss_mask), 0.1)
                all_acc.append(acc)

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(self.midibert.e2w):
                #print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                y[i] = y[i][:, ...].permute(0, 2, 1)

            # calculate losses
            losses, n_tok = [], []
            for i, etype in enumerate(self.midibert.e2w):
                n_tok.append(len(self.midibert.e2w[etype]))

                losses.append(self.compute_loss(y[i], ori_seq_batch_data[..., i], loss_mask))
                    
            # Bar chroma loss
            n_tok.append(1)
            losses.append(self.compute_chroma_loss(y[4], bar_chroma_gt, loss_chroma_mask))
            total_chroma_losses += losses[-1].item()

            total_loss_all = [x*y for x, y in zip(losses, n_tok)]
            total_loss = sum(total_loss_all)/sum(n_tok)   # weighted

            total_acc = [sum(x) for x in zip(total_acc, all_acc)]

            # update only in train
            if train:
                self.model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)
                self.optim.step()

            # acc
            accs = list(map(float, all_acc))

            if batch_id % 200 == 0 and train == True:
                print('Loss: {:.4f} | loss: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f} | acc: {:.3f}, {:.3f}, {:.3f}, {:.3f} \r'.format(
                    total_loss, *losses, *accs)) 

            losses = list(map(float, losses))
            total_losses += total_loss.item()
        
        total_chroma_losses = total_chroma_losses / len(training_data)
        print ('Avg chroma loss:', total_chroma_losses)
        return round(total_losses/len(training_data),3), [round(x.item()/len(training_data),3) for x in total_acc]

    def save_checkpoint(self, epoch, best_acc, valid_acc, 
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.midibert.state_dict(),
            'mask_lm': self.model.mask_lm.state_dict(),
            'best_acc': best_acc,
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'optimizer' : self.optim.state_dict()
        }

        torch.save(state, filename)

        best_mdl = filename.split('.')[0]+'_best.ckpt'
        if is_best:
            shutil.copyfile(filename, best_mdl)

