# M2BERT

This is the official implementation of the paper:

Jun-You Wang and Li Su, "Improving BERT for symbolic music understanding using token denoising and pianoroll prediction," accepted at ISMIR 2025.

The source code is modified from MidiBERT ( [https://github.com/wazenmai/MIDI-BERT](https://github.com/wazenmai/MIDI-BERT) ). In addition to using new backbone model, , new objective functions, and more training data, we also add source code for 12 downstream tasks, i.e., the SMCBenchmark (I put it in another repo; please see https://github.com/york135/SMCBenchmark ).

## Resources (Pretrained model and processed data)

You can download them from [https://github.com/york135/M2BERT_files](https://github.com/york135/M2BERT_files)

## Fine-tuning

Please see `script/finetune_modernbert_script.sh`.

```
bash script/finetune_modernbert_script.sh [ckpt_dir] \
 [note_data_root] [seq_data_root]
```

where `ckpt_dir` is the directory to the pretrained model checkpoint; `[note_data_root]` is the directory to the note-level SMC tasks for fine-tuning; `[seq_data_root]` is the directory to the sequence-level SMC tasks for fine-tuning.

If you download all the resources from york135/M2BERT_files, this should be:

```
bash script/finetune_modernbert_script.sh \
 ../M2BERT_files/M2BERT_pretrained_model \
 ../M2BERT_files/finetune_note_1024 \
 ../M2BERT_files/finetune_seq
```



**The following sections are still under construction**

## Training from scratch

### Prepare all the pretraining datasets

The processed npy files can be simply obtained here.

For those who want to reproduce the whole data preparation process:

For ASAP, pop1k7, pop909, pianist8, emopia, please follow the instruction of MidiBERT to put them into the correct directory (it should look like this one).

<img src="file:///C:/Users/User/AppData/Roaming/marktext/images/2025-06-06-17-05-58-image.png" title="" alt="" width="184">

For LMD, just download the lmd_matched dataset from and put it into '../lmd_matched'. 

Then, run:

```
bash script/prepare_pretrain_data.sh
```

That will take several hours. The total size of output dataset npy files is around 10GB. 

### Prepare all the fine-tuning datasets

Again, it is strongly recommended to simply obtain the processsed npy files here.

It is also fairly simple to use the unquantized SMCBenckmark data to create the fine-tuning datasets. First, put all the SMCBenchmark data at '../SMC_dataset', then run:

f