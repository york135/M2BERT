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
    ### path ###
    parser.add_argument('--dict', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--smc_dir', type=str)
    ### parameter ###
    # This value should be specified so that we can perform segmentation for note-level tasks
    parser.add_argument('--max_len', type=int)
    
    ### output ###    
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    return args



def main(): 
    args = get_args()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = CP(dict=args.dict)

    note_dir = os.path.join(args.smc_dir, 'downstream_note')
    if os.path.isdir(note_dir):
        note_dataset_list = ['augnet_train.npy', 'augnet_valid.npy', 'augnet_test.npy',
            'augnet_train.npy', 'augnet_valid.npy', 'augnet_test.npy',
            'pm2s_train.npy', 'pm2s_valid.npy', 'pm2s_test.npy',
            'pm2s_train.npy', 'pm2s_valid.npy', 'pm2s_test.npy',
            'pop909_train.npy', 'pop909_valid.npy',  'pop909_test.npy',
            'pop909_train.npy', 'pop909_valid.npy',  'pop909_test.npy',
            'orch_train.npy', 'orch_valid.npy', 'orch_test.npy',
            'tnua_train.npy', 'tnua_valid.npy', 'tnua_test.npy',
            'bps_motif_fold_0.npy', 'bps_motif_fold_1.npy', 'bps_motif_fold_2.npy',
            'bps_motif_fold_3.npy', 'bps_motif_fold_4.npy']

        note_groundtruth_list = ['augnet_train_chordrootans.npy', 'augnet_valid_chordrootans.npy', 
            'augnet_test_chordrootans.npy',
            'augnet_train_localkeyans.npy', 'augnet_valid_localkeyans.npy', 
            'augnet_test_localkeyans.npy',
            'pm2s_train_beatans.npy', 'pm2s_valid_beatans.npy', 'pm2s_test_beatans.npy',
            'pm2s_train_downbeatans.npy', 'pm2s_valid_downbeatans.npy', 'pm2s_test_downbeatans.npy',
            'pop909_train_melodyans.npy', 'pop909_valid_melodyans.npy',  'pop909_test_melodyans.npy',
            'pop909_train_velocityans.npy', 'pop909_valid_velocityans.npy',  'pop909_test_velocityans.npy',
            'orch_train_textureans.npy', 'orch_valid_textureans.npy', 'orch_test_textureans.npy',
            'tnua_train_violin_allans.npy', 'tnua_valid_violin_allans.npy', 'tnua_test_violin_allans.npy',
            'bps_motif_fold_0_mnidans.npy', 'bps_motif_fold_1_mnidans.npy', 'bps_motif_fold_2_mnidans.npy',
            'bps_motif_fold_3_mnidans.npy', 'bps_motif_fold_4_mnidans.npy']

        for note_dataset, note_groundtruth in zip(note_dataset_list, note_groundtruth_list):
            output_file = os.path.join(args.output_dir, note_dataset)
            ans_file = os.path.join(args.output_dir, note_groundtruth)

            if (os.path.isfile(output_file) and os.path.isfile(ans_file) 
                and args.overwrite is False):
                continue

            note_dataset_path = os.path.join(note_dir, note_dataset)
            note_groundtruth_path = os.path.join(note_dir, note_groundtruth)
            print (note_dataset_path, note_groundtruth_path)

            segments, ans = model.quantize_and_tokenize(note_dataset_path, note_groundtruth_path
                , max_len=args.max_len)

            np.save(output_file, segments)
            print(f'Data shape: {segments.shape}, saved at {output_file}')
            np.save(ans_file, ans)
            print(f'Answer shape: {ans.shape}, saved at {ans_file}')



if __name__ == '__main__':
    main()
