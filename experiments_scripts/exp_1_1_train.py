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
EXPERIMENT 1.1 SCRIPT (Classification)

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

        # Load sample size
        sample_size = config['FT_CONFIG']["sample_size"]
        hist_size = config['FT_CONFIG']["hist_size"]

        # Generate folders for the experiment
        if not os.path.exists("./experiments"):
                os.makedirs("./experiments")

        # Ensure model name directory exists
        model_dir = os.path.join("./experiments", config['GENERAL']['model_name'])
        if not os.path.exists(model_dir):
                os.makedirs(model_dir)

        # Ensure experiment title directory exists
        experiment_dir = os.path.join(model_dir, config['EXPERIMENT_CONFIG']['title'])
        if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

        # Create additional directories if they do not exist
        if not os.path.exists(config['FT_CONFIG']['ft_model_path']):
                os.makedirs(config['FT_CONFIG']['ft_model_path'])
        if not os.path.exists(config['FT_CONFIG']['ft_model_path']):
                os.makedirs(config['FT_CONFIG']['ft_modelH_path'])
        if not os.path.exists(config['EXPERIMENT_CONFIG']['results_dir']):
                os.makedirs(config['EXPERIMENT_CONFIG']['results_dir'])


        # General result dictionary
        results = {}
        
        # MODEL TRAINING
        if config['EXPERIMENT_CONFIG']['train']:

                print(f'01 - Training procedure: {sample_size} samples loaded')

                # Load dataset
                dataset = load_counterfact(config["FT_CONFIG"]["dataset_file"])

                if config["EXPERIMENT_CONFIG"]["paraphrases"]:

                        # Tokenize dataset
                        tokenized_dataset = dataset\
                                .map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=False), batched=True, batch_size=sample_size)\
                                .select_columns(['input_ids', 'attention_mask', 'labels', 'label_mask'])
                else:
                
                        # Tokenize dataset
                        tokenized_dataset = dataset\
                                .map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=False, rephrase=True), batched=True, batch_size=sample_size)\
                                .select_columns(['input_ids', 'attention_mask', 'labels', 'label_mask'])
                                
                # Create sub-sets
                dataset_t = tokenized_dataset.select(range(sample_size))

                # Load pt model
                pt_model = AutoModelForCausalLM.from_pretrained(config['GENERAL']['model_dir'])

                # Perform FT on A (trainer)
                score, _ = ft_trainer(
                        model=pt_model, 
                        lr=config['FT_CONFIG']["training_lr"], 
                        num_epochs=config['FT_CONFIG']["num_epochs"], 
                        batch_size=config['FT_CONFIG']["batch_size"], 
                        dataset_update=dataset_t, 
                        dataset_eval=[], 
                        out_dir = f"{config['FT_CONFIG']['ft_model_path']}", 
                        logging_dir=f"{config['FT_CONFIG']['logging_dir']}", 
                        historical_path=config['FT_CONFIG']["historical_file_path"], 
                        patterns=config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                        padding_mask=True,
                        pad_token=tokenizer.pad_token_id
                        )
                
                print(f'Accuracy dataset: {score}')

        # MODEL TRAINING
        if config['EXPERIMENT_CONFIG']['hist']:

                print(f'02 - Hisotrical data generation: {hist_size} samples loaded')

                # Load dataset
                dataset = load_counterfact("dataset/facts/counterfact_dataset/train.parquet")

                # Tokenize dataset
                tokenized_dataset = dataset.map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=False), batched=True, batch_size=sample_size+hist_size)
                        
                # Create sub-sets
                dataset_t = tokenized_dataset.select(range(sample_size,sample_size+hist_size))

                # Load pt model
                pt_model = AutoModelForCausalLM.from_pretrained(config['GENERAL']['model_dir'])

                # Perform FT on A (trainer)
                score, _ = ft_trainer(
                        model=pt_model, 
                        lr=config['FT_CONFIG']["training_lr"], 
                        num_epochs=config['FT_CONFIG']["num_epochs"], 
                        batch_size=config['FT_CONFIG']["batch_size"], 
                        dataset_update=dataset_t, 
                        dataset_eval=[], 
                        out_dir = f"{config['FT_CONFIG']['ft_modelH_path']}", 
                        logging_dir=f"{config['FT_CONFIG']['logging_dir']}", 
                        historical_path=config['FT_CONFIG']["gen_historical_file_path"], 
                        patterns=config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                        padding_mask=True,
                        pad_token=tokenizer.pad_token_id,
                        layer_norm=True
                        )
                
                print(f'Accuracy dataset: {score}')

if __name__ == "__main__":
    main()