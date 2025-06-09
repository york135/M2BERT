import os, csv
from pathlib import Path
import glob
import pickle
import pathlib
import argparse
import numpy as np
import pandas as pd
from data_creation.prepare_data.model import *


def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('-t', '--task', default='', choices=['melody', 'velocity', 'beat', 'downbeat'
        , 'texture', 'mnid', 'chordroot', 'localkey', 'violin_string', 'violin_position', 'violin_all'])

    ### path ###
    parser.add_argument('--dict', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--dataset', type=str, choices=['pop909', 's3', 'orch', 'augnet', 'pm2s', 'bps_motif', 'tnua'])

    ### parameter ###
    parser.add_argument('--max_len', type=int)
    
    ### output ###    
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    if (args.task == 'melody' or args.task == 'velocity') and args.dataset != 'pop909':
        print('[error] melody/velocity task is only supported for pop909 dataset')
        exit(1)
    elif (args.task == 'beat' or args.task == 'downbeat') and args.dataset != 'pm2s':
        print('[error] beat/downbeat task is only supported for pm2s dataset')
        exit(1)
    elif (args.task == 'chordroot' or args.task == 'localkey') and args.dataset != 'augnet':
        print('[error] chordroot/localkey task is only supported for augnet dataset')
        exit(1)
    elif (args.task == 'texture') and args.dataset != 's3' and args.dataset != 'orch':
        print('[error] texture task is only supported for s3 and orch dataset')
        exit(1)
    elif (args.task == 'mnid') and args.dataset != 'bps_motif':
        print('[error] MNID task is only supported for bps_motif dataset')
        exit(1)
    elif (args.task == 'violin_string' or args.task == 'violin_position' or args.task == 'violin_all') and args.dataset != 'tnua':
        print('[error] violin_string/violin_position/violin_all task is only supported for tnua dataset')
        exit(1)
    
    return args


def extract_pop909(files, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files)
    print(f'Number of {mode} files: {len(files)}') 

    segments, ans = model.prepare_finetune_pop909_data(files, args.task, max_len=args.max_len)

    dataset = args.dataset

    output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task}ans.npy')
    np.save(ans_file, ans)
    print(f'Answer shape: {ans.shape}, saved at {ans_file}')


def extract_bps_motif(dataset_root, files_id, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files_id)
    print(f'Number of {mode} files: {len(files_id)}') 

    csv_paths = []
    dbeat_paths = []

    for indice in files_id:
        piece = str(indice+1).zfill(2)
        csv_note_path = os.path.join(dataset_root, 'notes_for_mnid', piece+'-1.csv')
        downbeat_csv_path = os.path.join(dataset_root, 'for_dbeats', str(indice+1), 'dBeats.xlsx')

        csv_paths.append(csv_note_path)
        dbeat_paths.append(downbeat_csv_path)

    segments, ans = model.prepare_finetune_bpsmotif_data(csv_paths, dbeat_paths, args.task, max_len=args.max_len)

    dataset = args.dataset

    output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task}ans.npy')
    np.save(ans_file, ans)
    print(f'Answer shape: {ans.shape}, saved at {ans_file}')

def extract_s3(dataset_root, files_id, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files_id)
    print(f'Number of {mode} files: {len(files_id)}')

    csv_paths = []
    dbeat_paths = []
    annotation_paths = []

    for movement in files_id:
        note_dir = os.path.join(dataset_root, movement, 'annotations', 'note')
        texture_dir = os.path.join(dataset_root, movement, 'annotations'
            , 'orchestral_texture')
        downbeat_csv_path = os.path.join(dataset_root, movement, 'annotations'
            , 'temporal_attributes', 'downbeat.csv')

        for instrument_csv in sorted(os.listdir(note_dir)):
            instrument_csv_path = os.path.join(note_dir, instrument_csv)
            annotation_csv_path = os.path.join(texture_dir, instrument_csv)

            csv_paths.append(instrument_csv_path)
            annotation_paths.append(annotation_csv_path)
            dbeat_paths.append(downbeat_csv_path)
            
    segments, ans = model.prepare_finetune_s3_data(csv_paths, annotation_paths
                        , dbeat_paths, args.task, max_len=args.max_len)

    dataset = args.dataset

    output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task}ans.npy')
    np.save(ans_file, ans)
    print(f'Answer shape: {ans.shape}, saved at {ans_file}')

def extract_orch(dataset_root, file_list, args, model, mode=''):

    assert len(file_list)
    print(f'Number of {mode} files: {len(file_list)}') 

    midi_paths = []
    annotation_paths = []
    dbeat_paths = []

    for i in range(len(file_list)):
        dbeats_csv_name = os.path.basename(file_list[i][0])[:-4] + '.csv'
        midi_path = os.path.join(dataset_root, file_list[i][0])
        annotation_path = os.path.join(dataset_root, file_list[i][1])
        downbeat_csv_path = os.path.join(dataset_root, 'dataset', 'dbeats', dbeats_csv_name)

        midi_paths.append(midi_path)
        annotation_paths.append(annotation_path)
        dbeat_paths.append(downbeat_csv_path)

        # print (midi_path, annotation_path, downbeat_csv_path)
            
    segments, ans = model.prepare_finetune_orch_data(midi_paths, annotation_paths
                        , dbeat_paths, args.task, max_len=args.max_len)

    dataset = args.dataset

    output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task}ans.npy')
    np.save(ans_file, ans)
    print(f'Answer shape: {ans.shape}, saved at {ans_file}')

def extract_augnet(files, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files)
    print(f'Number of {mode} files: {len(files)}') 

    mxl_paths = []
    annotation_paths = []

    for file in files:
        annotation_path = os.path.join('../chord_dataset/dataset', file[3], file[0] + '.tsv')
        mxl_path = os.path.join('../AugmentedNet', file[1])

        annotation_paths.append(annotation_path)
        mxl_paths.append(mxl_path)
            
    segments, ans = model.prepare_finetune_augnet_data(mxl_paths, annotation_paths, args.task, max_len=args.max_len)

    dataset = args.dataset

    output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task}ans.npy')
    np.save(ans_file, ans)
    print(f'Answer shape: {ans.shape}, saved at {ans_file}')

def extract_pm2s(files, args, model, mode=''):
    segments, ans = model.prepare_finetune_pm2s_data(files, args.task, max_len=args.max_len)
    dataset = args.dataset

    output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task}ans.npy')
    np.save(ans_file, ans)
    print(f'Answer shape: {ans.shape}, saved at {ans_file}')

def extract_tnua(files, args, model, mode=''):
    segments, ans = model.prepare_finetune_tnua_data(files, args.task, max_len=args.max_len)
    dataset = args.dataset

    # segments = np.concatenate((segments, segments, segments, segments), axis=0)
    # ans = np.concatenate((ans, ans, ans, ans), axis=0)

    output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task}ans.npy')
    np.save(ans_file, ans)
    print(f'Answer shape: {ans.shape}, saved at {ans_file}')

def main(): 
    args = get_args()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = CP(dict=args.dict)

    if args.dataset == 'pop909':
        dataset = 'pop909_processed'
        train_files = glob.glob(f'Data/Dataset/{dataset}/train/*.mid')
        valid_files = glob.glob(f'Data/Dataset/{dataset}/valid/*.mid')
        test_files = glob.glob(f'Data/Dataset/{dataset}/test/*.mid')

        extract_pop909(sorted(train_files), args, model, 'train')
        extract_pop909(sorted(valid_files), args, model, 'valid')
        extract_pop909(sorted(test_files), args, model, 'test')

    elif args.dataset == 'bps_motif':
        dataset_root = '../BPS_motif_clean'
        fold_lists = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]
            , [14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25], [26, 27, 28, 29, 30, 31]]

        print ([len(fold_list) for fold_list in fold_lists])
        for i in range(len(fold_lists)):
            extract_bps_motif(dataset_root, fold_lists[i], args, model, mode=f'fold_{str(i)}')

    elif args.dataset == 'orch':
        dataset_root = '../orchestraTextureClassification'
        metadata_file = os.path.join(dataset_root, 'dataset', 'new_metadata.csv')
        valid_data = [1, 5, 10]
        test_data = [0, 4, 9]

        song_list = []
        with open(metadata_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[1] != 'composer':
                    label_file = row[10]
                    midi_file = row[6]
                    song_list.append([midi_file, label_file])

        train_list, valid_list, test_list = [], [], []
        
        for i in range(len(song_list)):
            if i in valid_data:
                valid_list.append(song_list[i])
            elif i in test_data:
                test_list.append(song_list[i])
            else:
                train_list.append(song_list[i])

        extract_orch(dataset_root, train_list, args, model, 'train')
        extract_orch(dataset_root, valid_list, args, model, 'valid')
        extract_orch(dataset_root, test_list, args, model, 'test')

    elif args.dataset == 'augnet':
        # We use the annotation files processed by AugmentNet as the ground-truth
        # Since AugmentNet converted note list to a pianoroll-like form (w/ 1/32 note resolution),
        # we obtain the original note list from the mxl file
        csv_meta_path = '../chord_dataset/dataset/dataset_summary.tsv'
        song_name_list = []
        cur_song_id = 0

        with open(csv_meta_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row[1] != 'file':
                    song_name_list.append([row[1], row[3], row[4], row[5]])

        train_list, valid_list, test_list = [], [], []
        for i in range(len(song_name_list)):
            if song_name_list[i][3] == 'training':
                train_list.append(song_name_list[i])
            elif song_name_list[i][3] == 'validation':
                valid_list.append(song_name_list[i])
            elif song_name_list[i][3] == 'test':
                test_list.append(song_name_list[i])

        extract_augnet(sorted(train_list), args, model, 'train')
        extract_augnet(sorted(valid_list), args, model, 'valid')
        extract_augnet(sorted(test_list), args, model, 'test')

    elif args.dataset == 'pm2s':
        train_list, valid_list, test_list = [], [], []

        metadata = pd.read_csv('../PM2S/dev/metadata/metadata.csv')
        for i in range(metadata.shape[0]):
            feature_path = os.path.join('../features', os.path.basename(metadata['feature_file'].iloc[i]))
            split = metadata['split'].iloc[i]

            if split == 'train':
                train_list.append(feature_path)
            elif split == 'valid':
                valid_list.append(feature_path)
            elif split == 'test':
                test_list.append(feature_path)

        extract_pm2s(sorted(train_list), args, model, 'train')
        extract_pm2s(sorted(valid_list), args, model, 'valid')
        extract_pm2s(sorted(test_list), args, model, 'test')

    elif args.dataset == 'tnua':
        root_dir = '../TNUA_violin_fingering_dataset'
        # Cross-violinist, within-dataset: We consider a train-validationtest partition in the TNUA dataset where the excerpts and
        # violinists in the train, validation, and test sets are all nonoverlapping. Denoting the 14 excerpts as {P01, P02, · · · , P14}
        # and the 10 violinists as {V01, V02, · · · , V10}, we chose excerpts from P01 to P09 performed by V01, V02, V03, and
        # V04 as the training set (36 recordings in total), P10 and P11
        # performed by V05, V06, and V07 as the validation set (6
        # recordings in total), and P12, P13, and P14 performed by
        # V08, V09, and V10 as the test set (9 recordings in total).
        train_list, valid_list, test_list = [], [], []

        # Obtained from https://github.com/wayonbvc/Violin-Fingering-Generation-Through-Audio-Symbolic-Fusion/blob/main/Code/train.py
        TNUA_song_name_list = ["bach1","bach2","beeth1","beeth2_1","beeth2_2"
            ,"elgar","flower","mend1","mend2","mend3","mozart1","mozart2_1","mozart2_2","wind"]

        # Training: V{01~04} - P{01~09}
        for i in range(1, 5):
            for j in range(9):
                train_list.append(os.path.join(root_dir, f"vio{i}_{TNUA_song_name_list[j]}.csv"))

        # Validation: V{05~07} - P{10~11}
        for i in range(5, 8):
            for j in range(9, 11):
                valid_list.append(os.path.join(root_dir, f"vio{i}_{TNUA_song_name_list[j]}.csv"))

        # Test: V{08~10} - P{12~14}
        for i in range(8, 11):
            for j in range(11, 14):
                test_list.append(os.path.join(root_dir, f"vio{i}_{TNUA_song_name_list[j]}.csv"))

        extract_tnua(sorted(train_list), args, model, 'train')
        extract_tnua(sorted(valid_list), args, model, 'valid')
        extract_tnua(sorted(test_list), args, model, 'test')

if __name__ == '__main__':
    main()
