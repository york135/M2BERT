import argparse
import numpy as np
import random
import pickle
import os
import json
import torch
torch.set_float32_matmul_precision('medium')

from torch.utils.data import DataLoader
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    ### pre-train dataset ###
    parser.add_argument("--datasets", type=str, nargs='+', default=['pop909','pianist8', 'pop1k7'
                            , 'ASAP', 'emopia'])
    
    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mask_percent', type=float, default=0.3
        , help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)      # hidden state
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    
    ### cuda ###
    parser.add_argument('--dataset_dir', default="Data/Dataset")
    parser.add_argument("--cpu", action="store_true")   # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1], help="CUDA device ids")

    args = parser.parse_args()

    return args


def load_data(args, datasets):
    training_data = []
    root = args.dataset_dir
    total_token_count = 0

    for dataset in tqdm(datasets):
        data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)

        # print(f'   {dataset}: {data.shape}')
        cur_token_count = 0
        for i in range(len(data)):
            cur_token_count += len(data[i])
                
        print ('Dataset', dataset, '; token count:', cur_token_count)
        total_token_count += cur_token_count

    print ('Total token count:', total_token_count)


def main():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    classes = ['Bar', 'Position', 'Pitch', 'Duration']
    pad_word_np = np.array([e2w[etype]['%s <PAD>' % etype] for etype in classes], dtype=int)

    print("\nLoading Dataset", args.datasets)
    load_data(args, args.datasets)

if __name__ == '__main__':
    main()
