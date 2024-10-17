"""
Not super clean. This script was ran manually multiple times, for all seeds. 

Below are all the results:

A is locality/Old knowledge
B is reliability/New Knowledge(counterfactual)
Gen is generalization (generalization)

10 SAMPLES 

    46 -> A: 0.952 - B: 0.2 - Gen: 0.0 

    16 -> A: 0.745 - B: 0.3 - Gen: 0.3 

    5 -> A: 0.908 - B: 0.5 - Gen: 0.4 

    114 -> A: 0.948 - B: 0.0 - Gen: 0.0 

    1220 -> A: 0.903 - B: 0.2 - Gen: 0.2
 100 SAMPLES 

    46 -> A: 0.435 - B: 0.35 - Gen: 0.18 

    16 -> A: 0.470  - B: 0.35 - Gen: 0.14 

    5 -> A: 0.554  - B: 0.23 - Gen: 0.14 

    114 -> A:0.258  - B: 0.26 - Gen: 0.1 

    1220 -> A:0.437  - B: 0.31 - Gen: 0.19

     1000 SAMPLES 

    46 -> A: 0.0261 - B: 0.0015 - Gen: 0.005 

    16 -> A: 0.1906  - B: 0.193 - Gen: 0.083 

    5 -> A: 0.1808  - B: 0.2 - Gen: 0.084 

    114 -> A:0.1756  - B: 0.167 - Gen: 0.076 

    1220 -> A:0.187  - B: 0.24 - Gen: 0.085

"""

import sys
# custom path insertion
sys.path.insert(1, '.')

import utils
from utils import *
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import itertools

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
Evaluating ROME with Seed 1220 and sample size 1000 (sample_sizeS) which corresponds to the ft model below.
The memit version of this script was more "automated" (evaluate_MEMIT.py)
'''    

def main():
        # Set tokenizer
        tokenizer = AutoTokenizer.from_pretrained('models/pt_models/gpt2-small')
        tokenizer.pad_token = tokenizer.eos_token

        # Load sample size
        sample_sizeG = 2000
        sample_sizeS = 1000

        # Load dataset
        dataset = load_counterfact("dataset/facts/counterfact_dataset/train.parquet")
       
        seed = 1220

        shuffled_dataset=dataset.shuffle(seed=seed)

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

        ########################################################################################################
        # Pe-train mode evaluation
                                    
        # Load pt model
        modelROME = AutoModelForCausalLM.from_pretrained('experiments/ROME')
        modelB = AutoModelForCausalLM.from_pretrained('experiments/gpt2-small/experiment_3_1/ft_modelB')
        accelerator = Accelerator()

        # Verify known facts from A
        dataset_A, modelROME, modelB = accelerator.prepare(dataset_A, modelROME, modelB)
        _, known_facts = verify_acc(modelB, dataset_A, True, tokenizer.pad_token_id, True)
        
        print(len(known_facts))
        # Create new dataset A
        dataset_A_filtered = dataset_A.select(known_facts)

        dataset_A_filtered, dataset_B_not, dataset_gen= accelerator.prepare(dataset_A_filtered, dataset_B_not, dataset_gen)

        scoreB = verify_acc(modelB, dataset_B, True, tokenizer.pad_token_id)

        scoreA = verify_acc(modelROME, dataset_A_filtered, True, tokenizer.pad_token_id)
        scoreB_not = verify_acc(modelROME, dataset_B_not, True, tokenizer.pad_token_id)
        scoreGen = verify_acc(modelROME, dataset_gen, True, tokenizer.pad_token_id)

        print(f'ModelB Accuracy - B:{scoreB}')
        print(f'ROME Accuracy scores - A:{scoreA} NOT B:{scoreB_not} GEN:{scoreGen}')

        del modelROME
        del modelB
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
