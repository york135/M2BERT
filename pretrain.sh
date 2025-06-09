export PYTHONPATH="."

python3 M2BERT/main.py --name=m2bert_rc12_pr_reduced --max_seq_len 1024 \
 --epochs 150 --batch_size 12 --datasets pop909 pianist8 pop1k7 ASAP emopia \
 --data_root ../MIDI-BERT-CP_classical/Data/CP_data/pretrain