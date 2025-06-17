export PYTHONPATH="."

python3 M2BERT/finetune.py --task=violin_all --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2021

python3 M2BERT/finetune.py --task=violin_all --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2022
 
python3 M2BERT/finetune.py --task=violin_all --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2023
 
python3 M2BERT/finetune.py --task=violin_all --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2024

python3 M2BERT/finetune_cross_val.py --task=mnid --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2

python3 M2BERT/finetune_cross_val.py --task=emotion --batch_size 12 --max_seq_len 512 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $3

python3 M2BERT/finetune_cross_val.py --task=composer --batch_size 12 --max_seq_len 512 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $3

python3 M2BERT/finetune.py --task=texture --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2

python3 M2BERT/finetune.py --task=melody --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2

python3 M2BERT/finetune.py --task=velocity --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2

python3 M2BERT/finetune.py --task=chordroot --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2

python3 M2BERT/finetune.py --task=localkey --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2

python3 M2BERT/finetune.py --task=beat --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2

python3 M2BERT/finetune.py --task=downbeat --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2

python3 M2BERT/finetune.py --task=genre --batch_size 12 --max_seq_len 512 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $3

python3 M2BERT/finetune_cross_val.py --task=mnid --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2022

python3 M2BERT/finetune_cross_val.py --task=emotion --batch_size 12 --max_seq_len 512 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $3 --seed 2022

python3 M2BERT/finetune_cross_val.py --task=composer --batch_size 12 --max_seq_len 512 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $3 --seed 2022

python3 M2BERT/finetune.py --task=texture --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2022

python3 M2BERT/finetune.py --task=melody --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2022

python3 M2BERT/finetune.py --task=velocity --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2022

python3 M2BERT/finetune.py --task=chordroot --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2022

python3 M2BERT/finetune.py --task=localkey --batch_size 6 --max_seq_len 1024 \
 --ckpt "$1/model_best.ckpt" --save_root "$1/" \
 --data_root $2 --seed 2022