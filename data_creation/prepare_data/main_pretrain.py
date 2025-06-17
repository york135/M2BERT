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

    ### path ###
    parser.add_argument('--dict', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--dataset', type=str, choices=["pop909", "pop1k7", "ASAP", "pianist8"
        , "emopia", "lmd"])

    ### output ###
    parser.add_argument('--dataset_dir', default="Data/Dataset")
    parser.add_argument('--output_dir', default="Data/CP_data/pretrain")

    args = parser.parse_args()
    return args


def extract(files, args, model):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files)

    print(f'Number of files: {len(files)}') 

    segments = model.prepare_pretrain_data(files)
    output_file = os.path.join(args.output_dir, f'{args.dataset}.npy')

    np.save(output_file, segments)
    print(f'Data shape: {len(segments)} {len(segments[0])} {len(segments[1])}, saved at {output_file}')

def main(): 
    args = get_args()
    dataset_dir = args.dataset_dir
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = CP(dict=args.dict)

    if args.dataset == 'pop909':
        dataset = 'pop909_processed'
    elif args.dataset == 'emopia':
        dataset = 'EMOPIA_1.0'
    elif args.dataset == 'pianist8':
        dataset = 'joann8512-Pianist8-ab9f541'

    if args.dataset == 'pop909' or args.dataset == 'emopia':
        train_files = glob.glob(os.path.join(dataset_dir, f'{dataset}/train/*.mid'))
        valid_files = glob.glob(os.path.join(dataset_dir, f'{dataset}/valid/*.mid'))
        test_files = glob.glob(os.path.join(dataset_dir, f'{dataset}/test/*.mid'))
        files = sorted(train_files + valid_files + test_files)

    elif args.dataset == 'pianist8':
        train_files = glob.glob(os.path.join(dataset_dir, f'{dataset}/train/*/*.mid'))
        valid_files = glob.glob(os.path.join(dataset_dir, f'{dataset}/valid/*/*.mid'))
        test_files = glob.glob(os.path.join(dataset_dir, f'{dataset}/test/*/*.mid'))
        files = sorted(train_files + valid_files + test_files)

    elif args.dataset == 'lmd':
        files = glob.glob(os.path.join(dataset_dir, f'*/*/*/*/*.mid'))

    elif args.dataset == 'pop1k7':
        files = glob.glob(os.path.join(dataset_dir, f'Pop1K7/midi_transcribed/*/*.midi'))

    elif args.dataset == 'ASAP':
        files = pickle.load(open(os.path.join(dataset_dir,'ASAP_song.pkl'), 'rb'))
        files = [os.path.join(dataset_dir, f'asap-dataset/{file}') for file in files]

    else:
        print('not supported')
        exit(1)

    extract(sorted(files), args, model)

if __name__ == '__main__':
    main()
