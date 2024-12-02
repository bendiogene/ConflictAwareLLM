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
EXPERIMENT 1.1 SCRIPT (Classification with output only (no internal/hidden state))

'''    

    
def get_prompt_probabilities(model, input_ids, attention_mask, tokenizer, num_endings=3):
    model.eval()
    probabilities = []  # full distributions for truncated prompts
    actual_token_probs = []  # probabilities of actual next tokens
    decoded_prompts = []
    
    # Get full sequence probabilities first
    with torch.no_grad():
        full_output = model(input_ids=input_ids, attention_mask=attention_mask)
    p_sro_final = torch.softmax(full_output.logits[:, -1, :], dim=-1).squeeze().cpu().numpy()
    
    # Get full sequence of tokens for reference
    full_tokens = input_ids[0].cpu().tolist()
    
    # Try different truncated prompt lengths
    for i in range(1, num_endings + 1):
        prompt_length = input_ids.size(1) - i
        prompt_input_ids = input_ids[:, :prompt_length]
        prompt_attention_mask = attention_mask[:, :prompt_length]
        
        prompt_text = tokenizer.decode(prompt_input_ids[0])
        decoded_prompts.append(prompt_text)
        
        with torch.no_grad():
            prompt_output = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask
            )
        
        probs = torch.softmax(prompt_output.logits[:, -1, :], dim=-1)
        probabilities.append(probs.squeeze().cpu().numpy())
        
        actual_next_token = full_tokens[prompt_length]
        actual_prob = probs[0, actual_next_token].item()
        actual_token_probs.append({
            'token_id': actual_next_token,
            'token': tokenizer.decode([actual_next_token]),
            'probability': actual_prob
        })
    
    # Add full sentence probability
    full_token_prob = {
        'token_id': full_tokens[-1],
        'token': tokenizer.decode([full_tokens[-1]]),
        'probability': p_sro_final[full_tokens[-1]]
    }
    
    return probabilities, decoded_prompts, actual_token_probs, p_sro_final, full_token_prob

def extract_classification_features(probs, prompts, actual_probs, p_sro_final, full_token_prob, tokenizer, k=100, n_bins=100):
    """
    Extract features F1, F2, and F3 from probability distributions
    
    Args:
        probs: List of probability distributions for truncated prompts
        prompts: List of decoded prompts
        actual_probs: List of dictionaries containing actual token probabilities
        p_sro_final: Probability distribution from full sentence
        full_token_prob: Probability of actual token in full sentence
        tokenizer: Tokenizer for vocabulary size
        k: Number of top values to consider for F2
        n_bins: Number of bins for distribution histograms in F3
    """
    vocab_size = len(tokenizer)
    num_endings = len(probs)
    
    # F1: Concatenate probabilities of actual tokens (including full sentence)
    f1 = np.array([ap['probability'] for ap in actual_probs] + [full_token_prob['probability']])
    
    # F2: Top-k values/indices for each distribution
    f2 = []
    target_indices_list = []
    
    # Process each truncated prompt distribution
    for prob_dist in probs:
        top_k_idx = np.argpartition(prob_dist, -k)[-k:]
        top_k_values = prob_dist[top_k_idx]
        sorted_indices = np.argsort(-top_k_values)
        top_k_idx = top_k_idx[sorted_indices]
        top_k_values = top_k_values[sorted_indices]
        normalized_indices = top_k_idx / vocab_size
        f2.extend(top_k_values)
        f2.extend(normalized_indices)
        target_indices_list.append(top_k_idx)
    
    # Add full sentence top-k features
    top_k_idx_final = np.argpartition(p_sro_final, -k)[-k:]
    top_k_values_final = p_sro_final[top_k_idx_final]
    sorted_indices = np.argsort(-top_k_values_final)
    top_k_idx_final = top_k_idx_final[sorted_indices]
    top_k_values_final = top_k_values_final[sorted_indices]
    normalized_indices_final = top_k_idx_final / vocab_size
    f2.extend(top_k_values_final)
    f2.extend(normalized_indices_final)
    target_indices_list.append(top_k_idx_final)
    
    f2 = np.array(f2)
    
    # F3: Binned distributions
    f3 = []
    
    # Process each truncated probability distribution
    for prob_dist in probs:
        hist, _ = np.histogram(prob_dist, bins=n_bins, range=(0, 1))
        f3.extend(hist / len(prob_dist))
    
    # Add full sentence distribution
    hist, _ = np.histogram(p_sro_final, bins=n_bins, range=(0, 1))
    f3.extend(hist / len(p_sro_final))
    
    # Create and add indicator vectors for actual tokens (including full sentence)
    for actual_prob in actual_probs + [full_token_prob]:
        indicator = np.zeros(vocab_size)
        indicator[actual_prob['token_id']] = 1.0
        hist, _ = np.histogram(indicator, bins=n_bins, range=(0, 1))
        f3.extend(hist / len(indicator))
    
    f3 = np.array(f3)
    
    f4 = np.concatenate([f1, f2, f3])
    return f1,f2,f3,f4, target_indices_list


def extract_features_output(data_type,  dataset_path, output_path, model, tokenizer, sample_size, save_stats=True):

    print(f"{data_type} facts extraction")

    # Initialize Accelerator
    accelerator = Accelerator()

    dataset = load_json(dataset_path)

    # Data preprocessing
    if data_type=='unknown':
        preprocessed_dataset = dataset
        sample_dataset = preprocessed_dataset 
    else:
        preprocessed_dataset = [preprocess_data(entry) for entry in dataset]
        sample_dataset = preprocessed_dataset[:sample_size]


    # Prepare training data
    training_data = []

    if data_type=='unknown':
        for entry in sample_dataset:

            training_data.extend(expand_entry_extraction(entry, True))
            # Tokenization and DataLoader setup
        tokenized_data = tokenize_data_extraction(training_data, tokenizer, True)
    else:
        for entry in sample_dataset:

            training_data.extend(expand_entry_extraction(entry, False)) 
            # Tokenization and DataLoader setup
        tokenized_data = tokenize_data_extraction(training_data, tokenizer, False) 

    dataloader = DataLoader(tokenized_data, 1, shuffle=False)
    
    num_training_steps = len(dataloader)

    progress_bar = tqdm(range(num_training_steps))
    
    # Extraction loop
    model.eval()
    feat_data = []
    ## Our framework assumes that we don't know the position of the fact, so we take the last token. To give this baseline an even better chance, we use the last 3 tokens (to make answers fit)
    num_endings_to_try=3

    # batch size is set to 1 in order to focus on a single fact
    for batch in dataloader:

        probs, prompts, actual_probs, p_sro_final, full_token_prob = get_prompt_probabilities(
            model,
            batch['input_ids'],
            batch['attention_mask'],
            tokenizer,
            num_endings_to_try
        )
        
        f1, f2, f3, f4, target_indices_list = extract_classification_features(probs, prompts, actual_probs, p_sro_final, full_token_prob, tokenizer, k=100, n_bins=100)
        # Create feature dictionary
        feat_dict = {
            'fact_id': batch['fact_id'][0],
            'class_id': batch['class_id'][0].item(),
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'f4': f4
        }
        
        feat_data.append(feat_dict)

        progress_bar.update(1)

    # File saving
    with open(output_path, 'wb') as file:
        pickle.dump(feat_data, file)
    
    print(f"FEATURES EXTRACTION COMPLETED: {len(feat_data)} rows have been generated")

    return None, None
 
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
                
                extract_features_output(data_type = 'familiar', 
                                                         dataset_path = dataset_path,
                                                         output_path = config['EXPERIMENT_CONFIG']['output_familiar_path'],
                                                         model = model,
                                                         tokenizer = tokenizer,
                                                         sample_size = sample_size)

                if config['EXPERIMENT_CONFIG']['unknown']:
                        extract_features_output(data_type = 'unknown', 
                                                                dataset_path = config['EXPERIMENT_CONFIG']['unknown_dataset_path'],
                                                                output_path = config['EXPERIMENT_CONFIG']['output_unknown_path'],
                                                                model = model,
                                                                tokenizer = tokenizer,
                                                                sample_size = sample_size)

                        concatenate_pickles(config['EXPERIMENT_CONFIG']['output_familiar_path'],
                                            config['EXPERIMENT_CONFIG']['output_unknown_path'],
                                            config['EXPERIMENT_CONFIG']['output_full_path'])

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
                        #There's no norm anymore
                        res['results'] = []
                        for i,seed in enumerate(config['EXPERIMENT_CONFIG']['seeds']):
                                seed_res = {}
                                X = []
                                y = np.array([])

                                # Extract features (f1, f2 etc) and labels (class_id) from the data
                                for item in data:
                                        print(item.keys())
                                        x = np.array([])
                                        for f in features:
                                                key = f"{f}"
                                                x=item[key].flatten()
                                                # for t in config['EXPERIMENT_CONFIG']['transformations']:
                                                #         x = np.concatenate((x,item[key][t].flatten()))

                                        X.append(x)

                                        y = np.concatenate((y,np.array([item['class_id']])))

                                X = np.array(X)

                                print(X.shape)

                                print(f'Fold n.{i}')
                                
                                print('SVM')
                                accuracy, f1, report, best_params = classification_results(X, y, seed, 'svm')
                                seed_res['svm'] = {'acc': accuracy, 'f1': f1, 'report': report, 'params': best_params}

                                print('RF')
                                accuracy, f1, report, best_params = classification_results(X, y, seed, 'rf')
                                seed_res['rf'] = {'acc': accuracy, 'f1': f1, 'report': report, 'params': best_params}

                                res['results'].append(seed_res)
                
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

