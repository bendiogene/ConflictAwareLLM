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

                
        # EXTRACTION PROCESS
        if config['EXPERIMENT_CONFIG']['extraction']:

                print(f'02 - Extraction procedure')

                if config['EXPERIMENT_CONFIG']['pt_model']:
                        model = AutoModelForCausalLM.from_pretrained(config['GENERAL']['model_dir'])
                        dataset_path =  config['EXPERIMENT_CONFIG']['dataset_known_path']
                else:
                        model = AutoModelForCausalLM.from_pretrained(config['FT_CONFIG']['ft_model_path'])
                        dataset_path =  config['EXPERIMENT_CONFIG']['dataset_path']
                
                grad_distr, act_distr = extract_features(data_type = 'familiar', 
                                                         patterns = config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                                                         dataset_path = dataset_path,
                                                         output_path = config['EXPERIMENT_CONFIG']['output_familiar_path'],
                                                         model = model,
                                                         tokenizer = tokenizer,
                                                         sample_size = sample_size,
                                                         transformations = config['EXPERIMENT_CONFIG']['transformations'],
                                                         norm_type = config['EXPERIMENT_CONFIG']['norm_type'],
                                                         hist_path = config['FT_CONFIG']['gen_historical_file_path'],
                                                         last_token = config['EXPERIMENT_CONFIG']['last_token'])

                t_grad = grad_distr[0]
                t_act = act_distr[0]

                f_grad = grad_distr[1]
                f_act = act_distr[1]

                if config['EXPERIMENT_CONFIG']['unknown']:
                        grad_distr, act_distr = extract_features(data_type = 'unknown', 
                                                                patterns = config['HISTORICAL_FEATURES_CONFIG']['patterns'],
                                                                dataset_path = config['EXPERIMENT_CONFIG']['unknown_dataset_path'],
                                                                output_path = config['EXPERIMENT_CONFIG']['output_unknown_path'],
                                                                model = model,
                                                                tokenizer = tokenizer,
                                                                sample_size = sample_size,
                                                                transformations = config['EXPERIMENT_CONFIG']['transformations'],
                                                                norm_type = config['EXPERIMENT_CONFIG']['norm_type'],
                                                                hist_path = config['FT_CONFIG']['gen_historical_file_path'],
                                                                last_token = config['EXPERIMENT_CONFIG']['last_token'])

                        u_grad = grad_distr[1]
                        u_act = act_distr[1]

                        concatenate_pickles(config['EXPERIMENT_CONFIG']['output_familiar_path'],
                                            config['EXPERIMENT_CONFIG']['output_unknown_path'],
                                            config['EXPERIMENT_CONFIG']['output_full_path'])

                else:
                        u_grad = np.array([])
                        u_act = np.array([])

                data_dict={'coherent_act':t_act, 'counterfact_act':f_act, 'unfamiliar_act':u_act, 'coherent_grad':t_grad, 'counterfact_grad':f_grad, 'unfamiliar_grad':u_grad}

                if config['EXPERIMENT_CONFIG']["save_stats"]:
                # Save the concatenated list to a new pickle file
                        with open(config['EXPERIMENT_CONFIG']["save_stats_path"], 'wb') as file:
                                pickle.dump(data_dict, file)



        # CLASSIFICATION PROCESS
        if config['EXPERIMENT_CONFIG']['classification']:

                output = []

                # load features file
                with open(config['EXPERIMENT_CONFIG']['output_full_path'], 'rb') as file:
                        data = pickle.load(file)

                for features in config['EXPERIMENT_CONFIG']['features']:
                        print(features)
                        res = {}

                        res['features'] = features

                        res['results'] = {}

                        for norm in config['EXPERIMENT_CONFIG']['norm_list']:

                                res['results'][norm] = []

                                for i,seed in enumerate(config['EXPERIMENT_CONFIG']['seeds']):

                                        seed_res = {}

                                        X = []
                                        y = np.array([])

                                        # Extract features (activations) and labels (class_id) from the data
                                        for item in data:
                                                x = np.array([])
                                                for f in features:

                                                        if norm is not None:
                                                                key = f"{f}_{norm}"
                                                        else:
                                                                key = f"{f}"

                                                        for t in config['EXPERIMENT_CONFIG']['transformations']:
                                                                x = np.concatenate((x,item[key][t].flatten()))

                                                X.append(x)

                                                y = np.concatenate((y,np.array([item['class_id']])))

                                        X = np.array(X)

                                        print(X.shape)

                                        print(f'Fold n.{i} - Norm {norm}')
                                        
                                        print('SVM')
                                        accuracy, f1, report, best_params = classification_results(X, y, seed, 'svm')
                                        seed_res['svm'] = {'acc': accuracy, 'f1': f1, 'report': report, 'params': best_params}

                                        print('RF')
                                        accuracy, f1, report, best_params = classification_results(X, y, seed, 'rf')
                                        seed_res['rf'] = {'acc': accuracy, 'f1': f1, 'report': report, 'params': best_params}

                                        res['results'][norm].append(seed_res)
                        
                        output.append(res)

                # Get the current timestamp and format it
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Create the filename with the timestamp
                filename = os.path.join(config['EXPERIMENT_CONFIG']['results_dir'], f'experiment_1_1_{timestamp}.json')

                # Step 3: Write the dictionary to a JSON file
                with open(filename, 'w') as json_file:
                        json.dump(output, json_file, indent=4)

if __name__ == "__main__":
    main()