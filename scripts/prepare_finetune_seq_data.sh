input_dir="../example_midis"

export PYTHONPATH='.'

# For sequential tasks, always make the max_len to 512 for fair comparison
# For note-level tasks, set max_len to 1024

# melody
python3 data_creation/prepare_data/main_finetune_seq.py --dataset=pianist8 --task=composer --output_dir Data/CP_data/finetune_seq/

# emotion
python3 data_creation/prepare_data/main_finetune_seq.py --dataset=emopia --task=emotion --output_dir Data/CP_data/finetune_seq/

# Genre
python3 data_creation/prepare_data/main_finetune_seq.py --dataset=tagatraum --task=genre --output_dir Data/CP_data/finetune_seq/