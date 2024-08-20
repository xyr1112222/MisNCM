import numpy as np
import sys
import os

sys.path.append('../../../')
sys.path.append('../../../python_parser')

import json
import logging
import argparse
import warnings
import torch
import time
from model import Model
from run import TextDataset
from utils import set_seed
from python_parser.parser_folder import remove_comments_and_docstrings
from utils import Recorder
from attacker import Attacker
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")


    args = parser.parse_args()
    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')  # 读取model的路径
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        # 如果路径存在且有内容，则从checkpoint load模型
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin1')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())
        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1  # 只有一个label?
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)

    checkpoint_prefix = 'checkpoint-best-acc/model-0.02-100.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir),strict=False)
    model.to(args.device)

    # ## Load CodeBERT (MLM) model
    # codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    # tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    # codebert_mlm.to('cuda')

    ## Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    # Load original source codes
    source_codes = []
    idx_lists = []
    with open(args.eval_data_file) as f:
        for line in f:
            js = json.loads(line.strip())
            # code = ' '.join(js['func'].split())
            code = js['func']
            idx1 = js['idx']
            source_codes.append(code)
            idx_lists.append(idx1)

    is_success = 0
    total_cnt = 0
    id_not_match_idx = []
    previous_id = None  # 用于记录上一个id
    for index, example in enumerate(eval_dataset):
        code = source_codes[index]
        idx2 = idx_lists[index]

        # 模型推理...
        logits, preds = model.get_results([example], args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[1].item()
        print(f"True label: {true_label}, Predicted label: {orig_label}")
        total_cnt += 1

    # 判断模型预测是否正确...
        if orig_label == true_label:
            is_success += 1
    if total_cnt > 0:
        success_rate = is_success / total_cnt
    else:
        success_rate = 0

    print("Total success count:", is_success)
    print("Total count:", total_cnt)
    print("success rate:", success_rate)

if __name__ == '__main__':
    main()