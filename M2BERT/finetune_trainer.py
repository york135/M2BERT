import shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_

from M2BERT.finetune_model import TokenClassification, SequenceClassification


class FinetuneTrainer:
    def __init__(self, midibert, train_dataloader, valid_dataloader, test_dataloader, layer, 
                lr, class_num, hs, testset_shape, cpu, cuda_devices=None, model=None, SeqClass=False
                , label_counts=None, task=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        self.midibert = midibert
        self.SeqClass = SeqClass
        self.layer = layer
        self.task = task
        self.class_num = class_num

        print('   device:',self.device, '; task name:', self.task)

        if model != None:    # load model
            print('load a fine-tuned model')
            self.model = model.to(self.device)
        else:
            print('init a fine-tune model, sequence-level task?', SeqClass)
            if SeqClass:
                self.model = SequenceClassification(self.midibert, class_num, hs).to(self.device)
            else:
                self.model = TokenClassification(self.midibert, class_num, hs).to(self.device)

        if torch.cuda.device_count() > 1 and not cpu:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        # self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0, 1.0]).to(self.device)
        #     , reduction='none')

        if label_counts is not None:
            weight = 1.0 / torch.clip(torch.tensor(label_counts), min=1.0)
            weight = torch.clip(weight / torch.min(weight), max=10.0)
            self.loss_func = nn.CrossEntropyLoss(weight=weight.to(self.device), reduction='none')
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.testset_shape = testset_shape
    
    def compute_loss(self, predict, target, loss_mask, seq):
        loss = self.loss_func(predict, target)
        if not seq:
            loss = loss * loss_mask
            loss = torch.sum(loss) / torch.sum(loss_mask)
        else:
            loss = torch.sum(loss)/loss.shape[0]
        return loss

 
    def train(self, disable=False):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, 0, self.SeqClass, disable)
        return train_loss, train_acc

    def valid(self, disable=False):
        self.model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = self.iteration(self.valid_data, 1, self.SeqClass, disable)
        return valid_loss, valid_acc

    def test(self, disable=False):
        self.model.eval()
        with torch.no_grad():
            test_loss, test_acc, all_output = self.iteration(self.test_data, 2, self.SeqClass, disable)
        return test_loss, test_acc, all_output

    def iteration(self, training_data, mode, seq, disable):
        pbar = tqdm.tqdm(training_data, disable=disable)
        # print (self.class_num)

        total_acc, total_cnt, total_loss = 0, 0, 0
        # confusion = [[0, 0], [0, 0]]
        confusion = np.array([[0 for j in range(self.class_num - 1)] for i in range(self.class_num - 1)])

        if mode == 2: # testing
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        batch_count = 0 
        for x, y in pbar:  # (batch, 512, 768)
            # if mode == 0 and batch_count >= 1000:
            #     break
            batch_count = batch_count + 1
            batch = x.shape[0]
            x, y = x.to(self.device), y.to(self.device)     # seq: (batch, 512, 4), (batch) / token: , (batch, 512)
            # print (y, torch.max(y), torch.min(y))
            # avoid attend to pad word
            if not seq:
                # Attend all notes that are NOT padding
                attn = (x[:, :, 0] < 2).float().to(self.device)   # (batch,512)
            else:
                attn = torch.ones((batch, 512)).to(self.device)     # attend each of them

            y_hat = self.model.forward(x, attn, self.layer)     # seq: (batch, class_num) / token: (batch, 512, class_num)

            # get the most likely choice with max
            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(self.device)
            if mode == 2:
                all_output[cnt : cnt+batch] = output
                cnt += batch

            # Accuracy at 16th note level
            # This computation is time-consuming so I only use it in valid/test (mode=1 or 2)
            # For training, I approximate it using standard note-level accuracy
            if (self.task == 'chordroot' or self.task == 'localkey') and (mode == 1 or mode == 2):
                # Recreate barline
                x[:, 0, 0] = 0
                bar_flag = (x[:, :, 0] == 0)
                max_bar_flag_sum = torch.max(torch.sum(bar_flag, dim=1))
                bar_flag_cumsum = torch.cumsum(bar_flag, dim=1)

                # Create 16th note grid
                bar_gt_grid = torch.zeros((x.shape[0], max_bar_flag_sum * 16)).long().to(self.device)
                bar_pred_grid = torch.zeros((x.shape[0], max_bar_flag_sum * 16)).long().to(self.device)
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        start = (bar_flag_cumsum[i][j] - 1) * 16 + x[i, j, 1]
                        if x[i, j, 0] != self.model.midibert.bar_pad_word:
                            bar_gt_grid[i][start:] = y[i][j]
                            bar_pred_grid[i][start:] = output[i][j]

                grid_attn = (bar_gt_grid != 0).float()
                # Convert prediction to tatum-level (16th note) chord/localkey prediction
                acc = torch.sum((bar_gt_grid == bar_pred_grid).float() * grid_attn)
                total_acc += acc
                total_cnt += torch.sum(grid_attn).item()

            elif not seq and (self.task == 'texture'):
                # The experiments of Chu et al. excluded the no-role class for evaluation
                bar_flag = (x[:, :, 0] == 0) * (y != 0)
                acc = torch.sum((y == output).float() * attn * bar_flag)
                total_acc += acc
                total_cnt += torch.sum(attn * bar_flag).item()

            elif not seq:
                acc = torch.sum((y == output).float() * attn)
                total_acc += acc
                total_cnt += torch.sum(attn).item()

                # Required for F1-score
                if self.task == 'beat' or self.task == 'downbeat' or self.task == 'mnid':
                    confusion[0][0] += int(torch.sum(((y == 1) & (output == 1)).float() * attn))
                    confusion[0][1] += int(torch.sum(((y == 1) & (output == 2)).float() * attn))
                    confusion[1][0] += int(torch.sum(((y == 2) & (output == 1)).float() * attn))
                    confusion[1][1] += int(torch.sum(((y == 2) & (output == 2)).float() * attn))
            else:
                acc = torch.sum((y == output).float())
                total_acc += acc
                total_cnt += y.shape[0]

            # calculate losses
            if not seq:
                y_hat = y_hat.permute(0,2,1)

            if self.task == 'texture':
                loss = self.compute_loss(y_hat, y, attn * (y != 0), seq)
            else:
                loss = self.compute_loss(y_hat, y, attn, seq)
            total_loss += loss.item()

            # udpate only in train
            if mode == 0:
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

        if self.task == 'beat' or self.task == 'downbeat' or self.task == 'mnid':
            total_notes = confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1]
            test_accuracy = (confusion[0][0] + confusion[1][1]) / total_notes
            test_precision = (confusion[1][1]) / max((confusion[0][1] + confusion[1][1]), 0.0001)
            test_recall = (confusion[1][1]) / max((confusion[1][0] + confusion[1][1]), 0.0001)
            test_f1 = (2.0 * test_precision * test_recall) / max((test_precision + test_recall), 0.0001)

            if mode == 2:
                return round(total_loss/len(training_data),4), round(test_f1,4), all_output
            return round(total_loss/len(training_data),4), round(test_f1,4)

        if mode == 2:
            return round(total_loss/len(training_data),4), round(total_acc.item()/total_cnt,4), all_output
        return round(total_loss/len(training_data),4), round(total_acc.item()/total_cnt,4)
        

    def save_checkpoint(self, epoch, train_acc, valid_acc, 
                        valid_loss, train_loss, is_best, filename):
        if torch.cuda.device_count() > 1 and not cpu:
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(), 
                'valid_acc': valid_acc,
                'valid_loss': valid_loss,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'optimizer' : self.optim.state_dict()
            }
        else:
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(), 
                'valid_acc': valid_acc,
                'valid_loss': valid_loss,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'optimizer' : self.optim.state_dict()
            }
        
        if is_best:
            best_mdl = filename.split('.')[0]+'_best.ckpt'
            torch.save(state, best_mdl)