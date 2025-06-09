import os
from pathlib import Path
import glob
import pickle
import pathlib
import argparse
import numpy as np
from data_creation.prepare_data.model import *


def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('-t', '--task', default='', choices=['composer', 'emotion', 'genre'])

    ### path ###
    parser.add_argument('--dict', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--dataset', type=str, choices=["pianist8", "emopia", "tagatraum"])
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--input_file', type=str, default='')

    ### parameter ###
    # For sequential tasks, the max_len is always set to 512 for a fair comparison
    # parser.add_argument('--max_len', type=int, default=512)
    
    ### output ###    
    parser.add_argument('--output_dir', default="Data/CP_data/finetune_seq")
    parser.add_argument('--name', default="")   # will be saved as "{output_dir}/{name}.npy"

    args = parser.parse_args()
    
    if args.task == 'composer' and args.dataset != 'pianist8':
        print('[error] composer task is only supported for pianist8 dataset')
        exit(1)
    elif args.task == 'genre' and args.dataset != 'tagatraum':
        print('[error] genre task is only supported for tagatraum dataset')
        exit(1)
    elif args.task == 'emotion' and args.dataset != 'emopia':
        print('[error] emotion task is only supported for emopia dataset')
        exit(1)
    elif args.dataset == None and args.input_dir == None and args.input_file == None:
        print('[error] Please specify the input directory or dataset')
        exit(1)

    return args


def extract(files, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files)

    print(f'Number of {mode} files: {len(files)}') 

    segments, ans = model.prepare_finetune_seq_data(files, args.task)

    dataset = args.dataset

    output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task}ans.npy')
    np.save(ans_file, ans)
    print(f'Answer shape: {ans.shape}, saved at {ans_file}')

def repartition_5fold(train_files, valid_files, test_files, task):
    
    all_files = train_files + valid_files + test_files
    # print (all_files[:5])
    # print (len(all_files))

    all_files = sorted(all_files)

    fold_list = [[] for i in range(5)]

    if task == "composer":
        already_partitioned = [0 for i in range(len(Composer))]
        class_count = [0 for i in range(len(Composer))]

        # First pass: get class count
        for path in all_files:
            name = path.split('/')[-2]
            class_count[Composer[name]] += 1
        # print (class_count)

        # Get 5 class-balanced folds, where the last fold may contain one less data
        for path in all_files:
            current_class_id = Composer[path.split('/')[-2]]
            to_this_fold = already_partitioned[current_class_id] * 5 // class_count[current_class_id]
            already_partitioned[current_class_id] += 1
            fold_list[to_this_fold].append(path)

    elif task == "emotion":
        already_partitioned = [0 for i in range(len(Emotion))]
        class_count = [0 for i in range(len(Emotion))]
        for path in all_files:
            name = path.split('/')[-1].split('_')[0]
            class_count[Emotion[name]] += 1

        for path in all_files:
            current_class_id = Emotion[path.split('/')[-1].split('_')[0]]
            to_this_fold = already_partitioned[current_class_id] * 5 // class_count[current_class_id]
            already_partitioned[current_class_id] += 1
            fold_list[to_this_fold].append(path)

    else:
        print ('???', task)

    return fold_list

def sample_train_val_files(train_files, valid_files):
    new_train_files = []
    new_valid_files = []

    label_counts = [0 for i in range(5)]
    train_files = sorted(train_files)
    for path in train_files:
        current_class_id = Genre[path.split('/')[-2]]
        if label_counts[current_class_id] < 500:
            new_train_files.append(path)
            label_counts[current_class_id] += 1

    label_counts = [0 for i in range(5)]
    valid_files = sorted(valid_files)
    for path in valid_files:
        current_class_id = Genre[path.split('/')[-2]]
        if label_counts[current_class_id] < 150:
            new_valid_files.append(path)
            label_counts[current_class_id] += 1

    return new_train_files, new_valid_files

def main(): 
    args = get_args()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = CP(dict=args.dict)

    if args.dataset == 'emopia':
        dataset = 'EMOPIA_1.0'
    elif args.dataset == 'pianist8':
        dataset = 'joann8512-Pianist8-ab9f541'

    # 5-fold CV for emopia, pianist8, and BPS-motif
    if args.dataset == 'emopia':
        # Re-partition the files! Ignore previous dataset partition......
        train_files = glob.glob(f'Data/Dataset/{dataset}/train/*.mid')
        valid_files = glob.glob(f'Data/Dataset/{dataset}/valid/*.mid')
        test_files = glob.glob(f'Data/Dataset/{dataset}/test/*.mid')

        print (len(train_files), len(valid_files), len(test_files))

    elif args.dataset == 'pianist8':
        train_files = glob.glob(f'Data/Dataset/{dataset}/train/*/*.mid')
        valid_files = glob.glob(f'Data/Dataset/{dataset}/valid/*/*.mid')
        test_files = glob.glob(f'Data/Dataset/{dataset}/test/*/*.mid')

    elif args.dataset == 'tagatraum':
        train_files = glob.glob(f'../tagatraum/train_clean/*/*.mid')
        valid_files = glob.glob(f'../tagatraum/valid_clean/*/*.mid')
        test_files = glob.glob(f'../tagatraum/test_clean/*/*.mid')

    else:
        print('not supported')
        exit(1)


    if args.dataset in {'emopia', 'pianist8'}:
        fold_lists = repartition_5fold(train_files, valid_files, test_files, task=args.task)
        print ([len(fold_list) for fold_list in fold_lists])
        for i in range(len(fold_lists)):
            extract(fold_lists[i], args, model, mode=f'fold_{str(i)}')

    elif args.dataset == 'tagatraum':
        train_files, valid_files = sample_train_val_files(train_files, valid_files)
        extract(train_files, args, model, mode='train')
        extract(valid_files, args, model, mode='valid')
        extract(test_files, args, model, mode='test')

if __name__ == '__main__':
    main()
