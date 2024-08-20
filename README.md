# Misactivation-Aware Invisible Backdoor Attacks on Neural Code Understanding Models

# Overview 
<img src="./figs/overview.png" alt="drawing" width="800">

# Environment Configuration

We use tree-sitter to parse code snippets and extract variable names. You need to go to `./parser`  folder and build tree-sitter using the following commands:

cd parser
bash build.sh

# Victim Models and Datasets

> <span style="color:red;"> If you cannot access to Google Driven in your region or countries, be free to email me and I will try to find another way to share the models.</span> 

## Datasets and Models

`model.bin` is a victim model obtained in our experiment (by fine-tuning models from [CodeBERT Repository](https://github.com/microsoft/CodeBERT)), and `model-poi.bin` is the backdoored model obtained in our experiment.Both models are in `..\code\saved_models directory`

The datasets are in `..\preprocess\dataset`.

The model and datasets can be downloaded from this  https://drive.google.com/drive/my-drive

## CodeBERT

_Fine-tune_
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base-mlm \
    --do_train \
    --train_data_file=../preprocess/dataset/train.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/test.jsionl \
    --epoch 5\
    --block_size 256 \
    --train_batch_size 12 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

_Attack_
cd preprocess
python get_sub.py \
    --store_path ./dataset/valid_subs.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/valid.jsonl \
    --block_size 512

_Trigger Generation_
python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base-mlm \
    --model_name_or_path=microsoft/codebert-base-mlm \
    --csv_store_path ./trigger.csv \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../preprocess/dataset/valid.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/valid.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee trigger.log

_Backdoored Model_
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base-mlm \
    --do_train \
    --train_data_file=../preprocess/dataset/train-poi.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/valid.jsionl \
    --epoch 4\
    --block_size 256 \
    --train_batch_size 12 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train-poi.log

_Evaluate_
python evaluate-asr.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base\
    --model_name_or_path=microsoft/codebert-base \
    --eval_data_file=../preprocess/dataset/test-poi.jsionl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee eval_.log




# Notes

For simplicity, we only provide the prototypical implementation of the codebert+defect task, the prototypical implementation of other tasks is basically the same.

