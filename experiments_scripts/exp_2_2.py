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
EXPERIMENT 2.2 SCRIPT (Lottery ticket)

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
        sample_sizeG = config['FT_CONFIG']["sample_size_1"]
        sample_sizeS = config['FT_CONFIG']["sample_size_2"]

        # Set-up results dict
        specs = {'title': config['EXPERIMENT_CONFIG']['title'],
                 'model_name': config['GENERAL']['model_name'],
                 'n_folds': len(config['EXPERIMENT_CONFIG']['seeds']),
                 'seeds': config['EXPERIMENT_CONFIG']['seeds'],
                 'n_rep': config['EXPERIMENT_CONFIG']['n_rep'],
                 'sample_sizeH':  config['FT_CONFIG']['sample_size_1'],
                 'sample_sizeA':  config['FT_CONFIG']['sample_size_2'],
                 'lr_H': config['FT_CONFIG']['training_lr_1'],
                 'lr_A': config['FT_CONFIG']['training_lr_2'],
                 'n_epochs_H': config['FT_CONFIG']['num_epochs_1'],
                 'n_epochs_A': config['FT_CONFIG']['num_epochs_2'],
                 'batch_size': config['FT_CONFIG']['batch_size']
                 }
        
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
        if not os.path.exists(config['FT_CONFIG']['ft_modelH_path']):
                os.makedirs(config['FT_CONFIG']['ft_modelH_path'])
        if not os.path.exists(config['FT_CONFIG']['ft_modelA_path']):
                os.makedirs(config['FT_CONFIG']['ft_modelA_path'])
        if not os.path.exists(config['EXPERIMENT_CONFIG']['results_dir']):
                os.makedirs(config['EXPERIMENT_CONFIG']['results_dir'])
        
        # General result dictionary
        results = {}

        # Load dataset
        dataset = load_counterfact(config["FT_CONFIG"]["dataset_file"])

        # Run different folds (each one with different seed)
        for f, seed in enumerate(config['EXPERIMENT_CONFIG']['seeds']):
                
                # Fold result dictionary
                fold_results = {}

                shuffled_dataset=dataset.shuffle(seed=seed)

                # Tokenize dataset
                tokenized_dataset = shuffled_dataset\
                        .map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=False), batched=True, batch_size=sample_sizeG+2*sample_sizeS)\
                        .select_columns(['input_ids', 'attention_mask', 'labels', 'label_mask'])

                print(f'Fold n. {f}')
                 
                # Create sub-sets
                dataset_H = tokenized_dataset.select(range(0, sample_sizeG))
                dataset_A = tokenized_dataset.select(range(sample_sizeG, sample_sizeG + sample_sizeS))

                print(f'DATASET H: {len(dataset_H)} samples loaded')
                print(f'DATASET A: {len(dataset_A)} samples loaded')

                # Set output directories
                out_dir_ftH = f"{config['FT_CONFIG']['ft_modelH_path']}"
                out_dir_ftA = f"{config['FT_CONFIG']['ft_modelA_path']}"
                
                
                ########################################################################################################
                # FT H

                print(f'Fold n. {f} - FTH')

                # Load pt model
                pt_model = AutoModelForCausalLM.from_pretrained(config['GENERAL']['model_dir'])

                # Perform FT on A (trainer)
                score_H, _ = ft_trainer(
                        model=pt_model, 
                        lr=config['FT_CONFIG']["training_lr_1"], 
                        num_epochs=config['FT_CONFIG']["num_epochs_1"], 
                        batch_size=config['FT_CONFIG']["batch_size"], 
                        dataset_update=dataset_H, 
                        dataset_eval=[], 
                        out_dir=out_dir_ftH, 
                        logging_dir=f"{config['FT_CONFIG']['logging_dir']}", 
                        historical_path=config['FT_CONFIG']["historical_file_path"], 
                        patterns=config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                        padding_mask=True,
                        pad_token=tokenizer.pad_token_id
                        )
                
                print(f'FTH scores - H:{score_H}')

                fold_results["fth"] = {"acc_H": score_H}

                del pt_model
                gc.collect()
                torch.cuda.empty_cache()

                ########################################################################################################
                # FT A

                print(f'Fold n. {f} - FTA')

                # Load pt model
                pt_model = AutoModelForCausalLM.from_pretrained(config['GENERAL']['model_dir'])

                # Perform FT on A (custom)
                epoch_score_A, _, score_A, _ = ft_custom(
                        model=pt_model, 
                        lr=config['FT_CONFIG']["training_lr_2"], 
                        num_epochs=config['FT_CONFIG']["num_epochs_2"], 
                        batch_size=config['FT_CONFIG']["batch_size"], 
                        dataset_update=dataset_A, 
                        dataset_eval=[], 
                        out_dir=None, 
                        logging_dir=f"{config['FT_CONFIG']['logging_dir']}",
                        padding_mask=True,
                        pad_token=tokenizer.pad_token_id,
                        save_model=False,
                        ev_epochs=config['EXPERIMENT_CONFIG']['ev_epochs']
                        )
                
                print(f'FTA scores - A:{score_A}')

                fold_results["fta"] = {"acc_A": score_A, 'epoch_acc_A': epoch_score_A}

                del pt_model
                gc.collect()
                torch.cuda.empty_cache()

                ########################################################################################################
                # C-FT A
                cfta_res = {}
                for t in config['HISTORICAL_FEATURES_CONFIG']["threshold"]:

                        cfta_res[t] = {}

                        strategies = config['EXPERIMENT_CONFIG']['strategies']

                        for s in strategies:

                                # C-FT C (norm busy)
                                print(f'Fold n. {f} - C-FTA ({s}) - threshold {t}')

                                res = {"epoch_acc_A":[], "acc_A": [],}

                                for r in range(config['EXPERIMENT_CONFIG']['n_rep']):

                                        # Load ft model
                                        pt_model = AutoModelForCausalLM.from_pretrained(config['GENERAL']['model_dir'])

                                        # Perform C-FT
                                        epoch_score_A, _, score_A, _, stubborn_number= ft_custom(
                                                model=pt_model, 
                                                lr=config['FT_CONFIG']["training_lr_2"], 
                                                num_epochs=config['FT_CONFIG']["num_epochs_2"], 
                                                batch_size=config['FT_CONFIG']["batch_size"], 
                                                dataset_update=dataset_A, 
                                                dataset_eval=[], 
                                                out_dir=None, 
                                                logging_dir=f"{config['FT_CONFIG']['logging_dir']}", 
                                                historical_path=config['FT_CONFIG'][strategies[s]['hist']], 
                                                gen_hist_path=config['FT_CONFIG'][strategies[s]['gen_hist']],
                                                patterns=config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                                                rnd=strategies[s]['rnd'],
                                                norm=strategies[s]['norm'],
                                                freeze=True,
                                                padding_mask=True,
                                                pad_token=tokenizer.pad_token_id,
                                                save_model=False,
                                                inverse=strategies[s]['inverse'],
                                                threshold=t,
                                                ev_epochs=config['EXPERIMENT_CONFIG']['ev_epochs']
                                                )

                                        print(f'Scores - A:{score_A}')

                                        res['acc_A'].append(score_A)
                                        res['epoch_acc_A'].append(epoch_score_A)

                                        del pt_model
                                        gc.collect()
                                        torch.cuda.empty_cache()

                                res['avg_accA'] = sum(res['acc_A'])/len(res['acc_A'])
                                res['number_stubborn'] = stubborn_number

                                cfta_res[t][s] = res

                fold_results['cfta'] = cfta_res
                results[f] = fold_results

        final_dict={'specs': specs, 'results': results}

        # Get the current timestamp and format it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the filename with the timestamp
        filename = os.path.join(config['EXPERIMENT_CONFIG']['results_dir'],
                                f'experiment_2_2_{timestamp}.json')

        # Step 3: Write the dictionary to a JSON file
        with open(filename, 'w') as json_file:
                json.dump(final_dict, json_file, indent=4)


if __name__ == "__main__":
    main()
