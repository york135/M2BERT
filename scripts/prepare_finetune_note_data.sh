input_dir="../example_midis"

export PYTHONPATH='.'

# For sequential tasks, always make the max_len to 512 for fair comparison
# For note-level tasks, set max_len to 1024

# MNID
# python3 data_creation/prepare_data/main_finetune_note.py \
# --dataset=bps_motif --task=mnid --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

# python3 data_creation/prepare_data/main_finetune_note.py \
#   --dataset=bps_motif --task=mnid_boundary --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

python3 data_creation/prepare_data/main_finetune_note.py \
  --dataset=bps_motif --task=motif_contrast --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

# # melody
# python3 data_creation/prepare_data/main_finetune_note.py \
#  --dataset=pop909 --task=melody --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

# # velocity
# python3 data_creation/prepare_data/main_finetune_note.py \
#  --dataset=pop909 --task=velocity --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

# python3 data_creation/prepare_data/main_finetune_note.py \
#  --dataset=s3 --task=texture --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

# python3 data_creation/prepare_data/main_finetune_note.py \
#   --dataset=orch --task=texture --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

# python3 data_creation/prepare_data/main_finetune_note.py \
#  --dataset=augnet --task=chordroot --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

#  python3 data_creation/prepare_data/main_finetune_note.py \
#  --dataset=augnet --task=localkey --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

# python3 data_creation/prepare_data/main_finetune_note.py \
#  --dataset=pm2s --task=beat --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

# python3 data_creation/prepare_data/main_finetune_note.py \
#  --dataset=pm2s --task=downbeat --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/

# python3 data_creation/prepare_data/main_finetune_note.py \
#  --dataset=tnua --task=violin_all --max_len 1024 --output_dir Data/CP_data/finetune_note_1024/