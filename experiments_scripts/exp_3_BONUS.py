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
import shutil
warnings.filterwarnings("ignore")

os.environ['CURL_CA_BUNDLE'] = ''


'''
EXPERIMENT 3.2 SCRIPT

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
        sample_sizeH = config['FT_CONFIG']["sample_size_1"]
        sample_sizeA = config['FT_CONFIG']["sample_size_2"]
        sample_sizeB = config['FT_CONFIG']["sample_size_3"]

        # Set-up results dict
        specs = {'title': config['EXPERIMENT_CONFIG']['title'],
                 'model_name': config['GENERAL']['model_name'],
                 'n_folds': config['EXPERIMENT_CONFIG']['n_folds'],
                 'n_rep': config['EXPERIMENT_CONFIG']['n_rep'],
                 'sample_sizeH':  config['FT_CONFIG']['sample_size_1'],
                 'sample_sizeA':  config['FT_CONFIG']['sample_size_2'],
                 'sample_sizeB':  config['FT_CONFIG']['sample_size_3'],
                 'lr_H': config['FT_CONFIG']['training_lr_1'],
                 'lr_A': config['FT_CONFIG']['training_lr_2'],
                 'lr_B': config['FT_CONFIG']['training_lr_2'],
                 'n_epochs_H': config['FT_CONFIG']['num_epochs_1'],
                 'n_epochs_A': config['FT_CONFIG']['num_epochs_2'],
                 'n_epochs_B': config['FT_CONFIG']['num_epochs_2'],
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

        if not os.path.exists(config['FT_CONFIG']['ft_modelB_path']):
                os.makedirs(config['FT_CONFIG']['ft_modelB_path'])

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
                tokenized_dataset_t = shuffled_dataset\
                        .map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=False), batched=True, batch_size=sample_sizeH+sample_sizeA+2*sample_sizeB)\
                        .select_columns(['input_ids', 'attention_mask', 'labels', 'label_mask']) 
                tokenized_dataset_c = shuffled_dataset\
                        .map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=True), batched=True, batch_size=sample_sizeH+sample_sizeA+2*sample_sizeB)\
                        .select_columns(['input_ids', 'attention_mask', 'labels', 'label_mask'])
                
                print(f'Fold n. {f}')
                 
                # Create sub-sets
                dataset_H = tokenized_dataset_t.select(range(0, sample_sizeH))
                dataset_A = tokenized_dataset_t.select(range(sample_sizeH, sample_sizeH + sample_sizeA))
                dataset_B = tokenized_dataset_t.select(range(sample_sizeH + sample_sizeA, sample_sizeH + sample_sizeA + sample_sizeB))
                dataset_B_not = tokenized_dataset_c.select(range(sample_sizeH + sample_sizeA, sample_sizeH + sample_sizeA + sample_sizeB))
                dataset_C = tokenized_dataset_t.select(range(sample_sizeH + sample_sizeA + sample_sizeB, sample_sizeH + sample_sizeA + 2*sample_sizeB))

                print(f'DATASET H: {len(dataset_H)} samples loaded')
                print(f'DATASET A: {len(dataset_A)} samples loaded')
                print(f'DATASET B: {len(dataset_B)} samples loaded')
                print(f'DATASET NOT B: {len(dataset_B_not)} samples loaded')

                # Set output directories
                out_dir_ftH = f"{config['FT_CONFIG']['ft_modelH_path']}"
                out_dir_ftA = f"{config['FT_CONFIG']['ft_modelA_path']}"
                out_dir_ftB = f"{config['FT_CONFIG']['ft_modelB_path']}"
                
                
                ########################################################################################################
                # FT H

                print(f'Fold n. {f} - FT H')

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
                        historical_path=config['FT_CONFIG']["gen_historical_file_path"], 
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

                print(f'Fold n. {f} - FT A')

                # Load pt model
                # ft_modelH = AutoModelForCausalLM.from_pretrained(config['FT_CONFIG']['ft_modelH_path'])
                pt_model = AutoModelForCausalLM.from_pretrained(config['GENERAL']['model_dir'])

                # Perform FT on A (trainer)
                score_A, _ = ft_trainer(
                        # model=ft_modelH,
                        model=pt_model, 
                        lr=config['FT_CONFIG']["training_lr_1"], 
                        num_epochs=config['FT_CONFIG']["num_epochs_2"], 
                        batch_size=config['FT_CONFIG']["batch_size"], 
                        dataset_update=dataset_A, 
                        dataset_eval=[], 
                        out_dir=out_dir_ftA, 
                        logging_dir=f"{config['FT_CONFIG']['logging_dir']}", 
                        historical_path=config['FT_CONFIG']["historical_file_path"], 
                        patterns=config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                        padding_mask=True,
                        pad_token=tokenizer.pad_token_id,
                        save_model=True
                        )
                
                print(f'FTA scores - A:{score_A}')

                fold_results["fta"] = {"acc_A": score_A}

                # del ft_modelH
                del pt_model
                gc.collect()
                torch.cuda.empty_cache()
                
                ########################################################################################################
                # C-FT B
                print(f'Fold n. {f} - C-FT B')

                res = {"epoch_acc_A":[], "epoch_acc_B":[], "acc_A": [], "acc_B": []}

                # Load ft model
                ft_modelA = AutoModelForCausalLM.from_pretrained(config['FT_CONFIG']['ft_modelA_path'])

                # Perform C-FT
                epoch_score_B, epoch_score_A, score_B, score_A, stubborn_number= ft_custom(
                        model=ft_modelA, 
                        lr=config['FT_CONFIG']["training_lr_2"], 
                        num_epochs=config['FT_CONFIG']["num_epochs_2"], 
                        batch_size=config['FT_CONFIG']["batch_size"], 
                        dataset_update=dataset_B, 
                        dataset_eval=dataset_A, 
                        out_dir=out_dir_ftB, 
                        logging_dir=f"{config['FT_CONFIG']['logging_dir']}", 
                        historical_path=config['FT_CONFIG']['historical_file_path'], 
                        gen_hist_path=config['FT_CONFIG']['gen_historical_file_path'],
                        patterns=config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                        rnd=False,
                        norm=False,
                        freeze=True,
                        padding_mask=True,
                        pad_token=tokenizer.pad_token_id,
                        save_model=True,
                        inverse=True,
                        threshold=config['HISTORICAL_FEATURES_CONFIG']['threshold']
                        )

                print(f'Scores - A:{score_A} B:{score_B}')

                del ft_modelA
                gc.collect()
                torch.cuda.empty_cache()
 
                fold_results['cft-b'] = {'acc_A': score_A, 'acc_B': score_B}

                dst = os.path.join("experiments/edit_comparison/ft_modelB",f"{f}")

                if os.path.exists(dst):
                        shutil.rmtree(dst)  # Remove the destination directory and its contents

                shutil.copytree(out_dir_ftB, dst)
                
                ########################################################################################################
                # FT NOT-B
                print(f'Fold n. {f} - FT NOT-B')

                res = {"epoch_acc_A":[], "epoch_acc_B":[], "acc_A": [], "acc_B": []}

                for r in range(config['EXPERIMENT_CONFIG']['n_rep']):

                        # Load ft model
                        ft_modelB = AutoModelForCausalLM.from_pretrained(config['FT_CONFIG']['ft_modelB_path'])

                        # Perform FT
                        epoch_score_B, epoch_score_A, score_B, score_A = ft_custom(
                                model=ft_modelB, 
                                lr=config['FT_CONFIG']["training_lr_2"], 
                                num_epochs=config['FT_CONFIG']["num_epochs_2"], 
                                batch_size=config['FT_CONFIG']["batch_size"], 
                                dataset_update=dataset_B_not, 
                                dataset_eval=dataset_A, 
                                out_dir=None, 
                                logging_dir=f"{config['FT_CONFIG']['logging_dir']}",
                                padding_mask=True,
                                pad_token=tokenizer.pad_token_id,
                                save_model=False
                                )
                        
                        print(f'FTB scores - A:{score_A} B:{score_B}')

                        res['acc_A'].append(score_A)
                        res['acc_B'].append(score_B)
                        res['epoch_acc_A'].append(epoch_score_A)
                        res['epoch_acc_B'].append(epoch_score_B)

                        del ft_modelB
                        gc.collect()
                        torch.cuda.empty_cache()

                        res['avg_accA'] = sum(res['acc_A'])/len(res['acc_A'])
                        res['avg_accB'] = sum(res['acc_B'])/len(res['acc_B'])
                
                fold_results["ft-notb"] = res

                ########################################################################################################
                # C-FT NOT-B
                gen_res = {}

                strategies = config['EXPERIMENT_CONFIG']['strategies']

                for s in strategies:

                        print(f'Fold n. {f} - C-FT NOT-B ({s})')

                        res = {"epoch_acc_A":[], "epoch_acc_B":[], "acc_A": [], "acc_B": []}

                        for r in range(config['EXPERIMENT_CONFIG']['n_rep']):

                                # Load ft model
                                ft_modelB = AutoModelForCausalLM.from_pretrained(config['FT_CONFIG']['ft_modelB_path'])

                                # Perform C-FT
                                epoch_score_B, epoch_score_A, score_B, score_A, stubborn_number= ft_custom(
                                        model=ft_modelB, 
                                        lr=config['FT_CONFIG']["training_lr_2"], 
                                        num_epochs=config['FT_CONFIG']["num_epochs_2"], 
                                        batch_size=config['FT_CONFIG']["batch_size"], 
                                        dataset_update=dataset_B_not, 
                                        dataset_eval=dataset_A, 
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
                                        threshold=config['HISTORICAL_FEATURES_CONFIG']['threshold']
                                        )

                                print(f'Scores - A:{score_A} B:{score_B}')

                                res['acc_A'].append(score_A)
                                res['acc_B'].append(score_B)
                                res['epoch_acc_A'].append(epoch_score_A)
                                res['epoch_acc_B'].append(epoch_score_B)

                                del ft_modelB
                                gc.collect()
                                torch.cuda.empty_cache()

                        res['avg_accA'] = sum(res['acc_A'])/len(res['acc_A'])
                        res['avg_accB'] = sum(res['acc_B'])/len(res['acc_B'])
                        res['number_stubborn'] = stubborn_number

                        gen_res[s] = res

                fold_results['cft-notb'] = gen_res

                ########################################################################################################
                # C-FT C
                gen_res = {}

                strategies = config['EXPERIMENT_CONFIG']['strategies']

                for s in strategies:

                        print(f'Fold n. {f} - C-FTC ({s})')

                        res = {"epoch_acc_A":[], "epoch_acc_C":[], "acc_A": [], "acc_C": []}

                        for r in range(config['EXPERIMENT_CONFIG']['n_rep']):

                                # Load ft model
                                ft_modelB = AutoModelForCausalLM.from_pretrained(config['FT_CONFIG']['ft_modelA_path'])

                                # Perform C-FT
                                epoch_score_C, epoch_score_A, score_C, score_A, stubborn_number= ft_custom(
                                        model=ft_modelB, 
                                        lr=config['FT_CONFIG']["training_lr_2"], 
                                        num_epochs=config['FT_CONFIG']["num_epochs_2"], 
                                        batch_size=config['FT_CONFIG']["batch_size"], 
                                        dataset_update=dataset_C, 
                                        dataset_eval=dataset_A, 
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
                                        threshold=config['HISTORICAL_FEATURES_CONFIG']['threshold']
                                        )

                                print(f'Scores - A:{score_A} C:{score_C}')

                                res['acc_A'].append(score_A)
                                res['acc_C'].append(score_C)
                                res['epoch_acc_A'].append(epoch_score_A)
                                res['epoch_acc_C'].append(epoch_score_C)

                                del ft_modelB
                                gc.collect()
                                torch.cuda.empty_cache()

                        res['avg_accA'] = sum(res['acc_A'])/len(res['acc_A'])
                        res['avg_accC'] = sum(res['acc_C'])/len(res['acc_C'])
                        res['number_stubborn'] = stubborn_number

                        gen_res[s] = res


                fold_results['cft-c'] = gen_res
                results[f] = fold_results

        final_dict={'specs': specs, 'results': results}

        # Get the current timestamp and format it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the filename with the timestamp
        filename = os.path.join(config['EXPERIMENT_CONFIG']['results_dir'],
                                f'experiment_3_2_{timestamp}.json')

        # Step 3: Write the dictionary to a JSON file
        with open(filename, 'w') as json_file:
                json.dump(final_dict, json_file, indent=4)


if __name__ == "__main__":
    main()
