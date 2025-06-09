export PYTHONPATH='.'

python3 data_creation/prepare_data/main_pretrain.py --dataset=ASAP --output_dir Data/pretrain/
python3 data_creation/prepare_data/main_pretrain.py --dataset=pop1k7 --output_dir Data/pretrain/
python3 data_creation/prepare_data/main_pretrain.py --dataset=pop909 --output_dir Data/pretrain/
python3 data_creation/prepare_data/main_pretrain.py --dataset=pianist8 --output_dir Data/pretrain/
python3 data_creation/prepare_data/main_pretrain.py --dataset=emopia --output_dir Data/pretrain/
python3 data_creation/prepare_data/main_pretrain.py --dataset=lmd --output_dir Data/pretrain/