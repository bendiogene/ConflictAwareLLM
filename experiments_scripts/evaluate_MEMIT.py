import sys
# custom path insertion
sys.path.insert(1, '.')

import utils
from utils import *
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
# import argparse
# import itertools

from torch.utils.tensorboard import SummaryWriter
import gc
from datetime import datetime
import json
from accelerate import Accelerator
import warnings
import pickle
warnings.filterwarnings("ignore")

os.environ['CURL_CA_BUNDLE'] = ''

'''
EXPERIMENT Incremental editing with 3 MEMIT models (10,100 and 1000 facts)

'''

# Define the path for results
results_path = './experiments/MEMIT/results/memit_evaluation.json'

def calculate_avg_std():
    from statistics import mean, stdev
    from scipy.stats import hmean
    import json
    import pandas as pd
    
    # Read results from JSON
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Organize results by sample size (10, 100, 1000)
    results_by_size = {10: {"scoreA": [], "scoreB_not": [], "scoreGen": [], "harmonic_mean": []},
                       100: {"scoreA": [], "scoreB_not": [], "scoreGen": [], "harmonic_mean": []},
                       1000: {"scoreA": [], "scoreB_not": [], "scoreGen": [], "harmonic_mean": []}}

    for entry in data:
        size = entry['sample_sizeS']
        scoreA = entry['results']['scoreA']
        scoreB_not = entry['results']['scoreB_not']
        scoreGen = entry['results']['scoreGen']

        # Append individual scores
        results_by_size[size]['scoreA'].append(scoreA)
        results_by_size[size]['scoreB_not'].append(scoreB_not)
        results_by_size[size]['scoreGen'].append(scoreGen)

        # Calculate harmonic mean if none of the values are zero
        if scoreA > 0 and scoreB_not > 0 and scoreGen > 0:
            harmonic_mean_value = hmean([scoreA, scoreB_not, scoreGen])
        else:
            harmonic_mean_value = 0  # Set to 0 if any value is zero

        # Append harmonic mean
        results_by_size[size]['harmonic_mean'].append(harmonic_mean_value)

    # Calculate mean and std for each case
    summary = {}
    for size, results in results_by_size.items():
        summary[size] = {
            "avg_scoreA": f"{mean(results['scoreA']):.3f} ({stdev(results['scoreA']):.3f})",
            "avg_scoreB_not": f"{mean(results['scoreB_not']):.3f} ({stdev(results['scoreB_not']):.3f})",
            "avg_scoreGen": f"{mean(results['scoreGen']):.3f} ({stdev(results['scoreGen']):.3f})",
            "avg_harmonic_mean": f"{mean(results['harmonic_mean']):.3f} ({stdev(results['harmonic_mean']):.3f})"
        }

    # Convert summary to DataFrame and print
    df_summary = pd.DataFrame.from_dict(summary, orient='index')
    print(df_summary)

    # Optionally, write the DataFrame to CSV or another file if needed
    df_summary.to_csv('./experiments/MEMIT/results/memit_evaluation_summary.csv')


def write_results_to_file(seed, folder_num, sample_sizeS, scoreA, scoreB_not, scoreGen):
    # Create a result dictionary
    result = {
        "seed": seed,
        "folder_num": folder_num,
        "sample_sizeS": sample_sizeS,
        "results": {
            "scoreA": scoreA,
            "scoreB_not": scoreB_not,
            "scoreGen": scoreGen
        }
    }
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    # Append the result to a JSON file
    if not os.path.exists(results_path):
        with open(results_path, 'w') as f:
            json.dump([result], f, indent=4)
    else:
        with open(results_path, 'r+') as f:
            data = json.load(f)
            data.append(result)
            f.seek(0)
            json.dump(data, f, indent=4)

def main():
    # Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/pt_models/gpt2-small')
    tokenizer.pad_token = tokenizer.eos_token

    # Load sample size
    sample_sizeG = 2000

    # Load dataset
    dataset = load_counterfact("dataset/facts/counterfact_dataset/train.parquet")

    # Seeds and corresponding folder numbers
    seeds = [46, 16, 5, 114, 1220]
    folder_numbers = ['0', '1', '2', '3','4']

    for seed, folder_num in zip(seeds, folder_numbers):
        print(f"\nEvaluating for seed {seed} (folder {folder_num})")
        
        shuffled_dataset = dataset.shuffle(seed=seed)

        # MEMIT models evaluation
        memit_models = [
            (f'experiments/MEMIT/{folder_num}/MEMIT_sizeB_10', f'experiments/edit_comparison_gpt2-small/ft_modelB_10/{folder_num}',10),
            (f'experiments/MEMIT/{folder_num}/MEMIT_sizeB_100', f'experiments/edit_comparison_gpt2-small/ft_modelB_100/{folder_num}',100),
            (f'experiments/MEMIT/{folder_num}/MEMIT_sizeB_1000', f'experiments/edit_comparison_gpt2-small/ft_modelB_1000/{folder_num}',1000)
        ]

        for memit_model, ft_modelB_path, sample_sizeS in memit_models:
            # Tokenize dataset
            tokenized_dataset_t = shuffled_dataset.map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=False), batched=True, batch_size=sample_sizeG+sample_sizeS)
            tokenized_dataset_c = shuffled_dataset.map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=True), batched=True, batch_size=sample_sizeG+sample_sizeS)
            tokenized_dataset_gen = shuffled_dataset.map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=True, rephrase=True), batched=True, batch_size=sample_sizeG+sample_sizeS)

            # Create sub-sets
            dataset_A = tokenized_dataset_t.select(range(sample_sizeG))
            dataset_B = tokenized_dataset_t.select(range(sample_sizeG, sample_sizeG + sample_sizeS))
            dataset_B_not = tokenized_dataset_c.select(range(sample_sizeG, sample_sizeG + sample_sizeS))
            dataset_gen = tokenized_dataset_gen.select(range(sample_sizeG, sample_sizeG + sample_sizeS))

            print(f'DATASET A: {len(dataset_A)} samples loaded')
            print(f'DATASET B: {len(dataset_B)} samples loaded')
            print(f'DATASET NOT B: {len(dataset_B_not)} samples loaded')

            # Initialize accelerator
            accelerator = Accelerator()

            print(f"Evaluating MEMIT model: {memit_model} with ft_modelB: {ft_modelB_path}")

            # Load MEMIT and B models
            modelMEMIT = AutoModelForCausalLM.from_pretrained(memit_model)
            modelB = AutoModelForCausalLM.from_pretrained(ft_modelB_path)

            # Verify known facts from A
            dataset_A, modelMEMIT, modelB = accelerator.prepare(dataset_A, modelMEMIT, modelB)
            _, known_facts = verify_acc(modelB, dataset_A, True, tokenizer.pad_token_id, True)

            print(f"Number of known facts: {len(known_facts)}")
            
            # Create new dataset A filtered by known facts
            dataset_A_filtered = dataset_A.select(known_facts)

            dataset_A_filtered, dataset_B_not, dataset_gen = accelerator.prepare(dataset_A_filtered, dataset_B_not, dataset_gen)

            # Evaluate MEMIT and B models
            scoreB = verify_acc(modelB, dataset_B, True, tokenizer.pad_token_id)
            scoreA = verify_acc(modelMEMIT, dataset_A_filtered, True, tokenizer.pad_token_id)
            scoreB_not = verify_acc(modelMEMIT, dataset_B_not, True, tokenizer.pad_token_id)
            scoreGen = verify_acc(modelMEMIT, dataset_gen, True, tokenizer.pad_token_id)

            print(f'ModelB Accuracy - B: {scoreB}')
            print(f'MEMIT Accuracy scores - A: {scoreA} NOT B: {scoreB_not} GEN: {scoreGen}')

            write_results_to_file(seed, folder_num, sample_sizeS, scoreA, scoreB_not, scoreGen)

            # Clean up
            del modelMEMIT
            del modelB
            gc.collect()
            torch.cuda.empty_cache()
            accelerator.free_memory()

if __name__ == "__main__":
    # Define the path for results
    results_path = './experiments/MEMIT/results/memit_evaluation.json'
    main()
    # At the very end of the main function
    calculate_avg_std()

