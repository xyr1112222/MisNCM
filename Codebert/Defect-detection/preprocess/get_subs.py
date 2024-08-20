import json
import sys
import copy
import torch
import argparse
from tqdm import tqdm

sys.path.append('../../../')
sys.path.append('../../../python_parser')

# from attacker import
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings
from utils import is_valid_variable_name, _tokenize
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

# Define cosine similarity function


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_data_file", default="./test.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model", default="../../../codebert-base-mlm", type=str,
                        help="Base Model")
    parser.add_argument("--store_path", default="./top10_subs.jsonl ", type=str,
                        help="results")

    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--index", nargs='+',
                        help="Optional input sequence length after tokenization.", default=['0', '1'])
    parser.add_argument("--top_k", type=int, default=10, help="Number of top substitutes to select")
    args = parser.parse_args()
    eval_data = []
    # Load model and tokenizer
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')

    # Load potential substitutes from output_with_scores.jsonl
    with open(args.eval_data_file) as rf:
        for i, line in enumerate(rf):
            if i < int(args.index[0]) or i >= int(args.index[1]):
                continue
            item = json.loads(line.strip())
            eval_data.append(item)
    print(len(eval_data))
    potential_substitutes = set()
    with open("output_with_scores.jsonl", "r") as vf:
        for line in vf:
            try:
                variables_data = json.loads(line.strip())
                if "variables" in variables_data and isinstance(variables_data["variables"], list):
                    potential_substitutes.update(variables_data["variables"])
            except json.JSONDecodeError:
                print(f"Error decoding line: {line.strip()}")
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # Open the output file for writing
    with open(args.store_path, "w") as wf:
        for item in tqdm(eval_data):
            identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(item["func"], "c"), "c")
            processed_code = " ".join(code_tokens)

            words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)

            variable_names = []
            for name in identifiers:
                if ' ' in name[0].strip():
                    continue
                variable_names.append(name[0])

            sub_words = [tokenizer_mlm.cls_token] + sub_words[:args.block_size - 2] + [tokenizer_mlm.sep_token]

            input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)]).to('cuda')

            # Compute embeddings for original variable names



            # Initialize the substitutes key in the item
            if "substitutes" not in item:
                item["substitutes"] = {}

            for var_name in variable_names:
                if not is_valid_variable_name(var_name, lang='c'):
                    continue
                top_candidates = []
                sims = []

                var_name_tokens = tokenizer_mlm.tokenize(var_name)
                ori_var_name_ids_ids = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(var_name_tokens)]).to('cuda')
                with torch.no_grad():
                    orig_embeddings = codebert_mlm.roberta(ori_var_name_ids_ids)[0]

                # Compute cosine similarity with potential substitutes
                for candidate in potential_substitutes:
                    candidate_tokens = tokenizer_mlm.tokenize(candidate)
                    substitute_ids = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(candidate_tokens)]).to('cuda')

                    with torch.no_grad():
                        new_embeddings = codebert_mlm.roberta(substitute_ids)[0]
                        # 计算沿第二维（维度索引为 1）的均值
                        new_embeddings = torch.mean(new_embeddings, dim=1, keepdim=True)

                    orig_word_embed = orig_embeddings[0]
                    new_word_embed = new_embeddings[0]

                    similarity = cos(orig_word_embed, new_word_embed)
                    if torch.is_tensor(similarity) and similarity.numel() == 1:
                        sims.append((candidate, similarity.item()))
                    else:
                        # Handle case where similarity is not a valid single-element tensor
                        continue
                    sims.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = [candidate for candidate, sim in sims[:20]]
                # Add top candidates to substitutes
                item["substitutes"][var_name] = top_candidates

            # Write the modified item to the output file
            wf.write(json.dumps(item) + "\n")
if __name__ == "__main__":
    main()
