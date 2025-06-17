export PYTHONPATH="."

python3 M2BERT/main.py --save_dir=$1 --max_seq_len 1024 \
 --epochs 150 --batch_size 12 --datasets pop909 pianist8 pop1k7 ASAP emopia lmd \
 --data_root $2