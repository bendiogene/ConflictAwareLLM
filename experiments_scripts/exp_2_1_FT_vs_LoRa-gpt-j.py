import sys
# custom path insertion
sys.path.insert(1, '.')

import utils
from utils import *
from utils.utils_ft_gptj import *
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def print_gpu_memory():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))


'''
EXPERIMENT 2.1 SCRIPT (Incremental learning) (with LoRA as an additional baseline)

H => lr1
A => lr2
B => lr1 (this was initially a bad copy paste that led to a different lr than intended for learning B, compared to A*)
LoRAB=> lr2
CFTB => lr2 

*This unfortunate bug led, however, to somewhat interesting insights we discuss in `analysis/plots_exp_2_1_Addendum_LR_impact_illustration.ipynb`

* Note that, for "historical reasons", the script still trains the model first on a dataset H of 10,000 facts, which became, in the end, useful only for the next experiment of the "lottery ticket" hypothesis (Training on dataset H is used as a "simulation" to identify the "winning subnetworks", in which new knowledge is placed).  

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
                        .map(lambda facts: tokenize_function(facts, tokenizer, counterfacts=False), batched=True, batch_size=sample_sizeH+sample_sizeA+sample_sizeB)\
                        .select_columns(['input_ids', 'attention_mask', 'labels', 'label_mask'])

                print(f'Fold n. {f}')
                 
                # Create sub-sets
                dataset_H = tokenized_dataset.select(range(0, sample_sizeH))
                dataset_A = tokenized_dataset.select(range(sample_sizeH, sample_sizeH + sample_sizeA))
                dataset_B = tokenized_dataset.select(range(sample_sizeH + sample_sizeA, sample_sizeH + sample_sizeA + sample_sizeB))

                print(f'DATASET H: {len(dataset_H)} samples loaded')
                print(f'DATASET A: {len(dataset_A)} samples loaded')
                print(f'DATASET B: {len(dataset_B)} samples loaded')

                # Set output directories
                out_dir_ftH = f"{config['FT_CONFIG']['ft_modelH_path']}"
                out_dir_ftA = f"{config['FT_CONFIG']['ft_modelA_path']}"
                out_dir_ftB = f"{config['FT_CONFIG']['ft_modelB_path']}"
                
                
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
                        historical_path=config['FT_CONFIG']["gen_historical_file_path"], 
                        patterns=config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                        padding_mask=True,
                        pad_token=tokenizer.pad_token_id
                        )
                
                print(f'FTH scores - H:{score_H}')

                fold_results["fth"] = {"acc_H": score_H}

                del pt_model
                gc.collect()
                with torch.no_grad():
                        torch.cuda.empty_cache()
                ### Trying more aggressive clean up:
                for obj in gc.get_objects():
                        try:
                                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                        del obj
                        except:
                                pass
                gc.collect()
                with torch.no_grad():
                        torch.cuda.empty_cache()


                #######################################################################################################
                # FT A

                print(f'Fold n. {f} - FTA')

                # Load pt model
                # ft_modelH = AutoModelForCausalLM.from_pretrained(config['FT_CONFIG']['ft_modelH_path'])
                pt_model = AutoModelForCausalLM.from_pretrained(config['GENERAL']['model_dir'])

                # Perform FT on A (trainer)
                score_A, _ = ft_trainer(
                        # model=ft_modelH, 
                        model = pt_model,
                        lr=config['FT_CONFIG']["training_lr_2"], 
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
                with torch.no_grad():
                        torch.cuda.empty_cache()

                ########################################################################################################
                # Normalizaion of historical data

                # Open hist data
                with open(config['FT_CONFIG']["historical_file_path"], 'rb') as file:
                        data = pickle.load(file)

                # Open gen hist data
                with open(config['FT_CONFIG']["gen_historical_file_path"], 'rb') as file:
                        data_gen = pickle.load(file)

                # Perform hist normalization
                for layer in data['grad_hist_dict']['raw']:
                        data['grad_hist_dict']['norm'][layer] = data['grad_hist_dict']['raw'][layer] / data_gen['grad_hist_dict']['raw'][layer]

                # Save modified data
                with open(config['FT_CONFIG']["historical_file_path"], 'wb') as file:
                        pickle.dump(data, file)

                
                ########################################################################################################
                # FT B
                print(f'Fold n. {f} - FTB')

                ftb_res = {"epoch_acc_A":[], "epoch_acc_B":[], "acc_A": [], "acc_B": []}

                for r in range(config['EXPERIMENT_CONFIG']['n_rep']):

                        # Load ft model
                        ft_modelA = AutoModelForCausalLM.from_pretrained(config['FT_CONFIG']['ft_modelA_path'])

                        score_B, score_A = ft_trainer(
                                model=ft_modelA, 
                                lr=config['FT_CONFIG']["training_lr_1"], 
                                num_epochs=config['FT_CONFIG']["num_epochs_2"], 
                                batch_size=config['FT_CONFIG']["batch_size"], 
                                dataset_update=dataset_B, 
                                dataset_eval=dataset_A, 
                                out_dir=out_dir_ftB, 
                                logging_dir=f"{config['FT_CONFIG']['logging_dir']}", 
                                historical_path=config['FT_CONFIG']["historical_file_pathB"], 
                                patterns=config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                                padding_mask=True,
                                pad_token=tokenizer.pad_token_id,
                                save_model=True
                                )
                        print(f'FTB scores - A:{score_A} B:{score_B}')

                        ftb_res['acc_A'].append(score_A)
                        ftb_res['acc_B'].append(score_B)
                        # zi: not needed anymore since moved to classic trainer
                        #ftb_res['epoch_acc_A'].append(epoch_score_A)
                        #ftb_res['epoch_acc_B'].append(epoch_score_B)

                        del ft_modelA
                        gc.collect()
                        with torch.no_grad():
                                torch.cuda.empty_cache()

                        ftb_res['avg_accA'] = sum(ftb_res['acc_A'])/len(ftb_res['acc_A'])
                        ftb_res['avg_accB'] = sum(ftb_res['acc_B'])/len(ftb_res['acc_B'])
                
                fold_results["ftb"] = ftb_res

                # ########################################## LoRA ########################################################
                ## Add LoRA Finetuning baseline => This is just to get a baseline comparison so no need to save model nor to get historical features
                # FT model A on dataset B USING LORA (instead of full finetuning)

                Loraftb_res = {"epoch_acc_A":[], "epoch_acc_B":[], "acc_A": [], "acc_B": []}

                for r in range(config['EXPERIMENT_CONFIG']['n_rep']):

                        # Load ft model
                        ft_modelA = AutoModelForCausalLM.from_pretrained(config['FT_CONFIG']['ft_modelA_path'])

                        # Perform FT
                        epoch_score_B, epoch_score_A, score_B, score_A= ft_custom_lora(
                                model=ft_modelA, 
                                lr=config['FT_CONFIG']["training_lr_2"], 
                                num_epochs=config['FT_CONFIG']["num_epochs_2"], 
                                batch_size=config['FT_CONFIG']["batch_size"], 
                                dataset_update=dataset_B, 
                                dataset_eval=dataset_A, 
                                out_dir=None, 
                                logging_dir=f"{config['FT_CONFIG']['logging_dir']}",
                                padding_mask=True,
                                pad_token=tokenizer.pad_token_id,
                                save_model=False
                                )
                        
                        print(f'LoRA FTB scores - A:{score_A} B:{score_B}')

                        Loraftb_res['acc_A'].append(score_A)
                        Loraftb_res['acc_B'].append(score_B)
                        Loraftb_res['epoch_acc_A'].append(epoch_score_A)
                        Loraftb_res['epoch_acc_B'].append(epoch_score_B)

                        del ft_modelA
                        gc.collect()
                        with torch.no_grad():
                                torch.cuda.empty_cache()

                        Loraftb_res['avg_accA'] = sum(Loraftb_res['acc_A'])/len(Loraftb_res['acc_A'])
                        Loraftb_res['avg_accB'] = sum(Loraftb_res['acc_B'])/len(Loraftb_res['acc_B'])
                
                fold_results["LoRA-ftb"] = Loraftb_res

                ########################################################################################################

        final_dict={'specs': specs, 'results': results}

        # Get the current timestamp and format it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the filename with the timestamp
        filename = os.path.join(config['EXPERIMENT_CONFIG']['results_dir'],
                                f'experiment_2_1_paper_version_LoRAvsFT_{timestamp}.json')

        # Step 3: Write the dictionary to a JSON file
        with open(filename, 'w') as json_file:
                json.dump(final_dict, json_file, indent=4)


if __name__ == "__main__":
    main()
