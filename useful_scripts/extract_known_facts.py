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
import pandas as pd

os.environ['CURL_CA_BUNDLE'] = ''


'''
EXPERIMENT 2.3 SCRIPT (Incremental learning)

'''    

def main():
        # Set config file
        parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
        parser.add_argument('--config', type=str, required=True, help='Path to the Yaml configuration file.')
        args = parser.parse_args()
        config_path = args.config
        config = load_configuration(config_path)

        # Device settings
        os.environ["CUDA_DEVICE_ORDER"] = config['DEVICE_CONFIG']['cuda_device_order']
        os.environ["CUDA_VISIBLE_DEVICES"] = config['DEVICE_CONFIG']['cuda_visible_devices']
        
        # Set tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['GENERAL']['model_dir'])
        tokenizer.pad_token = tokenizer.eos_token

       

        # Load dataset
        dataset = load_counterfact("dataset/facts/counterfact_dataset/train.parquet")
        # Load sample size
        sample_size = len(dataset)

        # Tokenize dataset
        tokenized_dataset= dataset.map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=False), batched=True, batch_size=sample_size).select(range(0,sample_size))

        ########################################################################################################
        # Pe-train mode evaluation
                                    
        # Load pt model
        pt_model = AutoModelForCausalLM.from_pretrained(config['GENERAL']['model_dir'])
        accelerator = Accelerator()

        tokenized_dataset, pt_model = accelerator.prepare(tokenized_dataset, pt_model)

        _, known_facts = verify_acc(pt_model, tokenized_dataset, True, tokenizer.pad_token_id, True)

        print(f'{len(known_facts)} known facts')

        dataset_known = dataset.select(known_facts)

        with open('/home/MinnieMouse/project/epmem_edit/dataset/facts/counterfact.json') as file:
                dataset_json = json.load(file)

        dataset_json_filtered = []

        n = 0
        dataset_known = pd.DataFrame(dataset_known)
        for i,row in dataset_known.iterrows():

                k = i + n
                while row['case_id'] != dataset_json[k]['case_id']:
                        n += 1
                        k = i + n

        
                dataset_json_filtered.append(dataset_json[k])

        print(len(dataset_json_filtered))

        with open('/home/MinnieMouse/project/epmem_edit/dataset/facts/counterfact_preknown.json', 'w') as file:
                json.dump(dataset_json_filtered, file, indent=4)
        

        del pt_model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()