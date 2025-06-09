import argparse
import numpy as np
import pickle
import os
import random

import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader
import torch
from transformers import BertConfig, AutoConfig
from M2BERT.model import MidiModernBERT, MidiBert
from M2BERT.finetune_trainer import FinetuneTrainer
from M2BERT.finetune_dataset import FinetuneDataset

from matplotlib import pyplot as plt
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### mode ###
    parser.add_argument('--task', choices=['mnid', 'composer', 'emotion'], required=True)
    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--ckpt', type=str)

    parser.add_argument('--save_root', type=str, default='result/finetune_midibert/')
    parser.add_argument('--data_root', type=str, default='Data/CP_data/finetune_note_1024')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true")  # default: false

    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--save_ckpt', action="store_true")
    
    ### cuda ###
    parser.add_argument("--cpu", action="store_true") # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,], help="CUDA device ids")

    args = parser.parse_args()

    if args.task == 'composer':
        args.class_num = 8
    elif args.task == 'emotion':
        args.class_num = 4
    elif args.task == 'mnid':
        args.class_num = 3

    return args


def load_cv_data(dataset, task, data_root, train_folds, val_fold, test_fold):

    if dataset not in ['pianist8', 'emopia', 'bps_motif']:
        print(f'Dataset {dataset} not supported')
        exit(1)
    
    X_train = np.concatenate([np.load(os.path.join(data_root, f'{dataset}_fold_{train_fold}.npy'), allow_pickle=True) 
        for train_fold in train_folds], axis=0)
    X_val = np.load(os.path.join(data_root, f'{dataset}_fold_{val_fold}.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(data_root, f'{dataset}_fold_{test_fold}.npy'), allow_pickle=True)

    print('X_train: {}, X_valid: {}, X_test: {}'.format(X_train.shape, X_val.shape, X_test.shape))

    y_train = np.concatenate([np.load(os.path.join(data_root, f'{dataset}_fold_{train_fold}_{task}ans.npy'), allow_pickle=True) 
        for train_fold in train_folds], axis=0)
    y_val = np.load(os.path.join(data_root, f'{dataset}_fold_{val_fold}_{task}ans.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_root, f'{dataset}_fold_{test_fold}_{task}ans.npy'), allow_pickle=True)

    print('y_train: {}, y_valid: {}, y_test: {}'.format(y_train.shape, y_val.shape, y_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    # argument
    args = get_args()
    
    # set seed
    seed = args.seed
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu
    np.random.seed(seed)
    random.seed(seed)

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset") 
    if args.task == 'composer':
        dataset = 'pianist8'
        seq_class = True
        total_fold = 5
    elif args.task == 'emotion':
        dataset = 'emopia'
        seq_class = True
        total_fold = 5
    elif args.task == 'mnid':
        dataset = 'bps_motif'
        seq_class = False
        total_fold = 5

    average_test_acc = 0
    for fold in range(total_fold):
        test_fold = fold
        val_fold = (fold+1) % total_fold
        train_folds = []
        for i in range(total_fold):
            if i != test_fold and i != val_fold:
                train_folds.append(str(i))

        X_train, X_val, X_test, y_train, y_val, y_test = load_cv_data(dataset, args.task
            , data_root=args.data_root, train_folds=train_folds, val_fold=str(val_fold), test_fold=str(test_fold))

        trainset = FinetuneDataset(X=X_train, y=y_train)
        validset = FinetuneDataset(X=X_val, y=y_val) 
        testset = FinetuneDataset(X=X_test, y=y_test) 

        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        # print("   len of train_loader",len(train_loader))
        valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
        # print("   len of valid_loader",len(valid_loader))
        test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
        # print("   len of test_loader",len(test_loader))

        if args.ckpt == 'result/pretrain/default/model_best.ckpt':
            print("\nBuilding BERT model")
            configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                        position_embedding_type='relative_key_query',
                                        hidden_size=args.hs)
            midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
        else:
            print("\nBuilding ModernBERT model")
            configuration = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
            configuration.vocab_size = 10
            configuration.max_position_embeddings = 1024
            configuration.num_hidden_layers = 12
            configuration.attention_dropout = 0.1
            configuration.mlp_dropout = 0.1
            configuration.pad_token_id = -1
            # print (configuration)
            midibert = MidiModernBERT(bertConfig=configuration, e2w=e2w, w2e=w2e)

        best_mdl = ''
        if not args.nopretrain:
            best_mdl = args.ckpt
            print("   Loading pre-trained model from", best_mdl.split('/')[-1])
            checkpoint = torch.load(best_mdl, map_location='cpu')
            midibert.load_state_dict(checkpoint['state_dict'], strict=False)
        
        index_layer = int(args.index_layer)-13
        print("\nCreating Finetune Trainer using index layer", index_layer)
        trainer = FinetuneTrainer(midibert, train_loader, valid_loader, test_loader, index_layer, args.lr, args.class_num,
                                    args.hs, y_test.shape, args.cpu, args.cuda_devices, None, seq_class, task=args.task)
        
        
        print("\nTraining Start")
        save_dir = os.path.join(args.save_root, args.task + '_' + str(fold))
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'model.ckpt')
        print("   save model at {}".format(filename))

        best_acc, best_epoch = 0, 0
        bad_cnt = 0
        test_acc_at_best_val = 0

        with open(os.path.join(save_dir, 'log'), 'a') as outfile:
            outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
            for epoch in range(args.epochs):
                train_loss, train_acc = trainer.train(disable=True)
                valid_loss, valid_acc = trainer.valid(disable=True)
                test_loss, test_acc, _ = trainer.test(disable=True)

                is_best = valid_acc >= best_acc
                best_acc = max(valid_acc, best_acc)
                
                if is_best:
                    bad_cnt, best_epoch = 0, epoch
                    test_acc_at_best_val = test_acc
                else:
                    bad_cnt += 1
                
                if args.save_ckpt:
                    trainer.save_checkpoint(epoch, train_acc, valid_acc, 
                                            valid_loss, train_loss, is_best, filename)

                outfile.write('Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}\n'.format(
                    epoch+1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))
                
            average_test_acc += test_acc_at_best_val
            if args.task == 'mnid':
                print ('Best val F1:', best_acc, '; at this time, test F1:', test_acc_at_best_val)
            else:
                print ('Best val acc:', best_acc, '; at this time, test acc:', test_acc_at_best_val)

    average_test_acc = round(average_test_acc / total_fold, 4)
    if args.task == 'mnid':
        print ('Cross-validation ended, test set mean F1:', average_test_acc)
    else:
        print ('Cross-validation ended, test set mean acc:', average_test_acc)

if __name__ == '__main__':
    main()
