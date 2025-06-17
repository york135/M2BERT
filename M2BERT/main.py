import argparse
import numpy as np
import random
import pickle
import os
import json
import torch
torch.set_float32_matmul_precision('medium')

from torch.utils.data import DataLoader
from transformers import AutoConfig
from model import MidiModernBERT
from trainer import BERTTrainer
from midi_dataset import MidiDataset, collate_fn

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--save_dir', type=str, default='exp/')

    ### pre-train dataset ###
    parser.add_argument("--datasets", type=str, nargs='+', default=['pop909', 'pianist8', 'pop1k7'
                            , 'ASAP', 'emopia'])
    parser.add_argument('--data_root', type=str, default='Data/CP_data/pretrain')
    
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
    parser.add_argument("--cpu", action="store_true")   # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,], help="CUDA device ids")

    args = parser.parse_args()

    return args


def load_data(datasets, data_root):
    training_data = []

    for dataset in tqdm(datasets):
        data = np.load(os.path.join(data_root, f'{dataset}.npy'), allow_pickle=True)
        training_data = training_data + list(data)

    print('   > all training data:', len(training_data), len(training_data[0]), len(training_data[0][0]))

    random.shuffle(training_data)
    split = int(len(training_data)*0.85)
    X_train, X_val = training_data[:split], training_data[split:]

    weighted_X_train = []
    weighted_X_val = []

    for i in range(len(X_train)):
        for repeat in range(0, min(len(X_train[i]), 512*1000), 512):
            weighted_X_train.append(X_train[i])

    for i in range(len(X_val)):
        for repeat in range(0, min(len(X_val[i]), 512*1000), 512):
            weighted_X_val.append(X_val[i])
            
    print('Weighted train/val data:', len(weighted_X_train), len(weighted_X_val))
    return weighted_X_train, weighted_X_val


def main():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    classes = ['Bar', 'Position', 'Pitch', 'Duration']
    pad_word_np = np.array([e2w[etype]['%s <PAD>' % etype] for etype in classes], dtype=int)

    print("\nLoading Dataset", args.datasets) 
    X_train, X_val = load_data(args.datasets, args.data_root)
    
    trainset = MidiDataset(X=X_train, pad_word_np=pad_word_np, max_seq_length=args.max_seq_len)
    validset = MidiDataset(X=X_val, pad_word_np=pad_word_np, max_seq_length=args.max_seq_len) 

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_fn, shuffle=True)
    print("   len of train_loader",len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    print("   len of valid_loader",len(valid_loader))

    print("\nBuilding ModernBERT model")
    configuration = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
    configuration.vocab_size = 10
    configuration.max_position_embeddings = 1024
    configuration.num_hidden_layers = 12
    configuration.attention_dropout = 0.1
    configuration.mlp_dropout = 0.1
    # configuration.window_size = [256, 256]
    # configuration.position_embedding_type ='relative_key_query'
    configuration.pad_token_id = -1
    midibert = MidiModernBERT(bertConfig=configuration, e2w=e2w, w2e=w2e)

    print("\nCreating BERT Trainer")
    trainer = BERTTrainer(midibert, train_loader, valid_loader, args.lr, args.batch_size
        , args.max_seq_len, args.mask_percent, args.cpu, args.cuda_devices, aux_output_dim=(12+86)*17)
    
    print("\nTraining Start")
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 30:
            print('valid acc not improving for 30 epochs')
            break
        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.valid()

        weighted_score = [x*y for (x,y) in zip(valid_acc, midibert.n_tokens)]
        avg_acc = sum(weighted_score)/sum(midibert.n_tokens)
        
        is_best = avg_acc > best_acc
        best_acc = max(avg_acc, best_acc)
        
        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1
        
        print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
            epoch+1, args.epochs, train_loss, train_acc, valid_loss, valid_acc))

        trainer.save_checkpoint(epoch, best_acc, valid_acc, 
                                valid_loss, train_loss, is_best, filename)

        with open(os.path.join(save_dir, 'log'), 'a') as outfile:
            outfile.write('Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n'.format(
                epoch+1, train_loss, train_acc, valid_loss, valid_acc))


if __name__ == '__main__':
    main()
