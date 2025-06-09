from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class MidiDataset(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """
    def __init__(self, X, pad_word_np, max_seq_length):
        self.data = X 
        self.max_seq_length = max_seq_length
        self.pad_word_np = torch.tensor(pad_word_np)

    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        # Somehow try to tell collate_fn max_seq_length
        return [torch.tensor(self.data[index]),
                 self.max_seq_length, self.pad_word_np]


def padding(data, pad_word, max_len):
    pad_len = max_len - len(data)
    to_pad = torch.repeat_interleave(pad_word.unsqueeze(0), repeats=pad_len, dim=0)
    data = torch.cat((data, to_pad), dim=0)
    return data

def collate_fn(data):
    # data, bar pos, max size, pad token
    mixed_data = []
    max_seq_length = data[0][1]

    # Randomly get a max_seq_length-token segment; if too short, then pad it
    for i in range(len(data)):
        if data[i][0].shape[0] <= max_seq_length:
            mixed_data.append(padding(data[i][0], data[i][2], max_len=data[i][1]).unsqueeze(0))
        else:
            start_time = int(torch.randint(low=0, high=data[i][0].shape[0] - max_seq_length, size=(1,)))
            cur_data = data[i][0][start_time:start_time+max_seq_length]
            mixed_data.append(cur_data.unsqueeze(0))
    mixed_data = torch.cat(mixed_data)

    pad_token = data[i][2]
    bar_chroma_gt = torch.full((mixed_data.shape[0], mixed_data.shape[1], 17*(12+86)), fill_value=0).float()
    for i in range(len(mixed_data)):
        cur_bin = torch.zeros((17,12+86)).float()
        next_bin = torch.zeros((17,12+86)).float()
        start_note = 0
        for j in range(len(mixed_data[i])):
            # is a bar start, "flush" the information of the previous bar
            if mixed_data[i][j][0] == 0 and j > start_note:
                # Get tatum-level groundtruth
                cur_bin_select = torch.index_select(cur_bin, 0, mixed_data[i,start_note:j,1])
                cur_bin_select = cur_bin_select / torch.clip(torch.sum(cur_bin_select, dim=1, keepdim=True), min=0.1)
                cur_bin_select = cur_bin_select * 2.0

                cur_bin = cur_bin / torch.clip(torch.sum(cur_bin, dim=1, keepdim=True), min=0.1)
                # The same note is counted twice in the denominator...
                cur_bin = (cur_bin[:16,:] * 2.0).reshape(-1)
                cur_bin = torch.repeat_interleave(cur_bin.unsqueeze(0), repeats=(j - start_note), dim=0)
                cur_bin = torch.cat((cur_bin, cur_bin_select), dim=1)

                bar_chroma_gt[i][start_note:j] = cur_bin
                # Renew
                cur_bin = next_bin
                next_bin = torch.zeros((17,12+86)).float()
                start_note = j

            if mixed_data[i][j][2] != pad_token[2]:
                # Add current note to the chroma & pitch distribution
                # Position: 0~15 (1/16 ~ 16/16, or, more correctly, 0/16 ~ 15/16)
                # Duration: 0~63 (1/32 ~ 64/32)
                # Duration dim: 3 (bar, position, pitch, duration)
                start = mixed_data[i][j][1]
                end = mixed_data[i][j][1] + int(round(float(mixed_data[i][j][3] + 1) / 2.0))

                # Chroma
                cur_bin[start:min(end, 16), mixed_data[i][j][2] % 12] += 1.0
                # Pitch value, index start from 12
                cur_bin[start:min(end, 16), 12 + mixed_data[i][j][2]] += 1.0

                if end > 16:
                    end = end - 16
                    next_bin[0:min(end, 16), mixed_data[i][j][2] % 12] += 1.0
                    next_bin[0:min(end, 16), 12 + mixed_data[i][j][2]] += 1.0

            if j == len(mixed_data[i]) - 1:
                cur_bin_select = torch.index_select(cur_bin, 0, mixed_data[i,start_note:j+1,1])
                cur_bin_select = cur_bin_select / torch.clip(torch.sum(cur_bin_select, dim=1, keepdim=True), min=0.1)
                cur_bin_select = cur_bin_select * 2.0

                cur_bin = cur_bin / torch.clip(torch.sum(cur_bin, dim=1, keepdim=True), min=0.1)
                cur_bin = (cur_bin[:16,:] * 2.0).reshape(-1)
                cur_bin = torch.repeat_interleave(cur_bin.unsqueeze(0), repeats=(j + 1 - start_note), dim=0)
                cur_bin = torch.cat((cur_bin, cur_bin_select), dim=1)

                bar_chroma_gt[i][start_note:j+1] = cur_bin
                # Renew
                cur_bin = next_bin
                next_bin = torch.zeros((17,12+86)).float()
                start_note = j+1

    return (mixed_data, bar_chroma_gt)
