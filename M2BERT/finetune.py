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
    parser.add_argument('--task', choices=['melody', 'velocity'
        , 'genre', 'beat', 'downbeat', 'chordroot', 'localkey'
        , 'violin_string', 'violin_position', 'violin_all', 'texture'], required=True)
    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--ckpt', type=str)
    # parser.add_argument('--ckpt', default='result/pretrain/default/model_best.ckpt')
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

    if args.task == 'melody':
        args.class_num = 4
    elif args.task == 'velocity':
        args.class_num = 7
    elif args.task == 'beat' or args.task == 'downbeat':
        args.class_num = 3
    elif args.task == 'chordroot':
        args.class_num = 36
    elif args.task == 'localkey':
        args.class_num = 38
    elif args.task == 'genre':
        args.class_num = 5
    elif args.task == 'violin_all':
        args.class_num = 241
    elif args.task == 'texture':
        args.class_num = 8

    return args


def load_data(dataset, data_root, task):

    if dataset not in ['pop909', 'tagatraum', 'pm2s', 'augnet', 'tnua', 'orch']:
        print(f'Dataset {dataset} not supported')
        exit(1)
        
    X_train = np.load(os.path.join(data_root, f'{dataset}_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_root, f'{dataset}_valid.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(data_root, f'{dataset}_test.npy'), allow_pickle=True)

    print('X_train: {}, X_valid: {}, X_test: {}'.format(X_train.shape, X_val.shape, X_test.shape))

    
    y_train = np.load(os.path.join(data_root, f'{dataset}_train_{task}ans.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(data_root, f'{dataset}_valid_{task}ans.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_root, f'{dataset}_test_{task}ans.npy'), allow_pickle=True)

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

    label_counts = None

    print("\nLoading Dataset") 
    if args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909' 
        seq_class = False
    elif args.task == 'beat' or args.task == 'downbeat':
        dataset = 'pm2s' 
        seq_class = False
    elif args.task == 'genre':
        dataset = 'tagatraum'
        seq_class = True
    elif args.task == 'chordroot' or args.task == 'localkey':
        dataset = 'augnet'
        seq_class = False
    elif args.task == 'violin_string' or args.task == 'violin_position' or args.task == 'violin_all':
        dataset = 'tnua'
        seq_class = False
    elif args.task == 'texture':
        dataset = 'orch'
        seq_class = False

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset, args.data_root, args.task)
    
    if args.task == 'beat' or args.task == 'downbeat':
        X_val = np.concatenate((X_val, X_train[-500:]), axis=0)
        y_val = np.concatenate((y_val, y_train[-500:]), axis=0)
        X_train = X_train[:-500]
        y_train = y_train[:-500]

    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val) 
    testset = FinetuneDataset(X=X_test, y=y_test) 

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)

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
        # configuration.window_size = [256, 256]
        # configuration.position_embedding_type ='relative_key_query'
        configuration.pad_token_id = -1
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
                                args.hs, y_test.shape, args.cpu, args.cuda_devices, None, seq_class, label_counts=label_counts
                                , task=args.task)
    
    
    print("\nTraining Start")
    save_dir = os.path.join(args.save_root, args.task)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0
    test_acc_at_best_val = 0

    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
        for epoch in range(args.epochs):
            train_loss, train_acc = trainer.train()
            valid_loss, valid_acc = trainer.valid()
            test_loss, test_acc, _ = trainer.test()

            is_best = valid_acc >= best_acc
            best_acc = max(valid_acc, best_acc)
            
            if is_best:
                bad_cnt, best_epoch = 0, epoch
                test_acc_at_best_val = test_acc
            else:
                bad_cnt += 1
            
            print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {} | Test loss: {} | Test acc: {}'.format(
                epoch+1, args.epochs, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))

            if args.save_ckpt:
                trainer.save_checkpoint(epoch, train_acc, valid_acc, 
                                        valid_loss, train_loss, is_best, filename)


            outfile.write('Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}\n'.format(
                epoch+1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))

    if args.task == 'beat' or args.task == 'downbeat':
        print ('Best val F1:', best_acc, '; at this time, test F1:', test_acc_at_best_val)
    elif args.task == 'violin_finger':
        print ('Best val macro-F1:', best_acc, '; at this time, test macro-F1:', test_acc_at_best_val)
    else:
        print ('Best val acc:', best_acc, '; at this time, test acc:', test_acc_at_best_val)

if __name__ == '__main__':
    main()
