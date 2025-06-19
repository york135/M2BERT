# M2BERT

This is the official implementation of the paper:

Jun-You Wang and Li Su, "Improving BERT for symbolic music understanding using token denoising and pianoroll prediction," accepted at ISMIR 2025.

The source code is modified from MidiBERT ( [https://github.com/wazenmai/MIDI-BERT](https://github.com/wazenmai/MIDI-BERT) ). In addition to using new backbone model, , new objective functions, and more training data, we also add source code for 12 downstream tasks, i.e., the SMCBenchmark (I put it in another repo; please see https://github.com/york135/SMCBenchmark ).

## Get started

```
pip install -r requirements.txt
```

## Resources (Pretrained model and processed data)

You can download them from [https://github.com/york135/M2BERT_files](https://github.com/york135/M2BERT_files)

Some of the files have to be unzipped.

## Fine-tuning

Please see `scripts/finetune_modernbert_script.sh`.

```
bash scripts/finetune_modernbert_script.sh [ckpt_dir] \
 [note_data_root] [seq_data_root]
```

where `ckpt_dir` is the directory to the pretrained model checkpoint; `[note_data_root]` is the directory to the note-level SMC tasks for fine-tuning; `[seq_data_root]` is the directory to the sequence-level SMC tasks for fine-tuning.

If you download all the resources from york135/M2BERT_files, this should be:

```
bash scripts/finetune_modernbert_script.sh \
 ../M2BERT_files/M2BERT_pretrained_model \
 ../M2BERT_files/finetune_note_1024 \
 ../M2BERT_files/finetune_seq
```

### Fine-tuning details

To strike a trade-off between minimizing randomness and evaluation time, for the 12 downstream tasks, we divide them into three types based on the scale of available datasets:

- High resource: SGC, BP, DbP (training dataset > 100MB)

- Mid resource: CR, LK, ME, VE, OTC, PS, ER, MNID (training dataset > 1MB, for 5-fold cross-validation tasks, we sum up the dataset size of three folds)

- Low resource: VF (training dataset < 1MB)

For high-resource tasks, I conduct fine-tuning only **once** per task (random seed: 2021); for mid-resource tasks, I conduct fine-tuning **twice** per task (random seed: 2021 and 2022); for low-resource tasks, I conduct fine-tuning **four times** per task (random seed: 2021, 2022, 2023, and 2024). The reported results are the average of all runs. 

This strategy allows us to reduce the randomness for low- and mid-resource tasks while maintaining the efficiency of the whole evaluation process (it takes around four hours to finish all 12 tasks on an Ada6000 GPU).

## Training from scratch

### Prepare pretraining datasets

The processed npy files can be simply obtained from [https://github.com/york135/M2BERT_files](https://github.com/york135/M2BERT_files)

For those who want to reproduce the whole data preparation process:

For **ASAP, pop1k7, pop909, pianist8, emopia**, please follow the instruction of MidiBERT (see [https://github.com/wazenmai/MIDI-BERT/tree/CP/data_creation](https://github.com/wazenmai/MIDI-BERT/tree/CP/data_creation)) to put them into the correct directory and perform pre-processing. It should look like:

```
[dataset_dir]
|-- pop909_processed
|-- EMOPIA_1.0
|-- joann8512-Pianist8-ab9f541
|-- asap-dataset
`-- Pop1K7
```

For **LMD**, just download the lmd_matched dataset from http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz and unzip it (assume you unzip it to `[lmd_dir]`). 

Then, run:

```
bash scripts/prepare_pretrain_data.sh [dataset_dir] [lmd_dir] \
  [output_data_root]
```

where `[output_data_root]` is the output directory of the pre-training datasets.

That will take several hours. The total size of output dataset npy files is around 10GB. 

### Prepare fine-tuning datasets

To convert the unquantized SMCBenckmark data to the CP tokens to create the fine-tuning datasets, run:

```
bash scripts/prepare_smc_benckmark.sh [SMC_directory] [output_data_root]
```

where `SMC_directory` is the directory to the SMCBenckmark (it should contain two folders: one for note-level tasks and another for sequence-level tasks); `[output_data_root]` is the output directory of the fine-tuning datasets (both `note_data_root` and `seq_data_root` will be the same in this case). 

### Run pre-training

For the reduced dataset (following MidiBERT), run:

```
bash pretrain.sh [save_dir] [data_root]
```

`save_dir` : the directory to save the pre-trained model

`data_root`: the directory to the pre-trained dataset (the `output_data_root` specified in `prepare_pretrain_data.sh`)

For the full dataset (including the lmd_aligned dataset), run:

```
bash pretrain_full.sh [save_dir] [data_root]
```

By the way, I only ran the full dataset pre-training for 25 epochs, so the pre-trained model checkpoint that I provide in the york135/M2BERT_files repo is the 25-epoch version. It is very welcomed if anyone who has very powerful GPU devices can provide the 150-epoch checkpoint (or even longer? I just don't know when will the full dataset pre-training be converged). I'm sure that will be much better than my 25-epoch one.

## Additional experiment results

Here are the ablation study results that we did not include in the ISMIR 2025 paper due to page limit issue.

**nochroma** here means that the pianoroll prediction objective only predicts the pianoroll representation but not chroma representation.

| Model  | Objectives                                     | Dataset | SGC  | BP   | DbP  | CR   | LK   | ME   |
|:------:|:----------------------------------------------:| ------- |:----:|:----:|:----:|:----:|:----:|:----:|
| M2BERT | $\text{RC}_{4, 12, 12}$                        | Reduced | .404 | .867 | .779 | .838 | .791 | .983 |
| M2BERT | $\text{RC}_{4, 12, 12}$+Pianoroll **nochroma** | Reduced | .419 | .869 | .765 | .840 | .796 | .981 |
| M2BERT | $\text{RC}_{4, 12, 12}$+Pianoroll (proposed)   | Reduced | .405 | .867 | .768 | .847 | .804 | .982 |

| Model  | Objectives                                     | Dataset | VE   | OTC  | PS   | ER   | VF   | MNID |
|:------:|:----------------------------------------------:|:-------:|:----:|:----:|:----:|:----:|:----:|:----:|
| M2BERT | $\text{RC}_{4, 12, 12}$                        | Reduced | .534 | .707 | .740 | .658 | .511 | .719 |
| M2BERT | $\text{RC}_{4, 12, 12}$+Pianoroll **nochroma** | Reduced | .535 | .703 | .744 | .674 | .523 | .724 |
| M2BERT | $\text{RC}_{4, 12, 12}$+Pianoroll (proposed)   | Reduced | .532 | .725 | .742 | .667 | .536 | .718 |
