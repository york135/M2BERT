export PYTHONPATH='.'

python3 data_creation/prepare_data/main_pretrain.py --dataset=ASAP \
		--dataset_dir $1 --output_dir $3
python3 data_creation/prepare_data/main_pretrain.py --dataset=pop1k7 \
		--dataset_dir $1 --output_dir $3
python3 data_creation/prepare_data/main_pretrain.py --dataset=pop909 \
		--dataset_dir $1 --output_dir $3
python3 data_creation/prepare_data/main_pretrain.py --dataset=pianist8 \
		--dataset_dir $1 --output_dir $3
python3 data_creation/prepare_data/main_pretrain.py --dataset=emopia \
		--dataset_dir $1 --output_dir $3
python3 data_creation/prepare_data/main_pretrain.py --dataset=lmd \
		--dataset_dir $2  --output_dir $3