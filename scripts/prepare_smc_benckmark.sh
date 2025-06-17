export PYTHONPATH='.'

# melody
python3 data_creation/prepare_data/quantize_smc_benchmark.py \
 --smc_dir $1 --max_len 1024 --output_dir $2 --overwrite
