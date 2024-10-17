import sys
# custom path insertion

sys.path.insert(1, '.')
import utils
from utils.parameters_hooks import *
import torch
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, AdamW, DataCollatorWithPadding
import pickle
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from numpy.random import default_rng
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

from joblib import parallel_backend



def extract_features(data_type, patterns, dataset_path, output_path, model, tokenizer, sample_size, transformations, norm_type, hist_path, last_token, save_stats=True):

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


    # Prepare optimizer
    optimizer = AdamW(model.parameters(), lr=0)

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
    
    # Prepare model, optimizer, and dataloader for acceleration
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    num_training_steps = len(dataloader)

    progress_bar = tqdm(range(num_training_steps))
    
    # Hooks definition
    hooks=ParametersHooks(model, patterns, history_path=hist_path, last_token=last_token)

    hooks.attach_hooks()

    # Extraction loop
    model.eval()
    global_step = 0
    feat_data = []

    act_distr = [np.array([]),np.array([])]
    grad_distr = [np.array([]),np.array([])]

    # batch size is set to 1 in order to focus on a single fact
    for batch in dataloader:
        optimizer.zero_grad()

        input_data = {
        'input_ids': batch['input_ids'],
        'attention_mask': batch['attention_mask'],
        'labels': batch['labels']
        }

        # Run the model to get the activations
        outputs = model(**input_data)

        # print(outputs)
        # Compute the loss
        # loss = outputs.loss
        loss = outputs['loss']
        

        # Backpropagation of the loss using the accelerator
        accelerator.backward(loss)

        # loss.backward()
        
        #last_token_idx = batch['answer_tokens_idx'][0].cpu().tolist()[0]-1
        last_token_idx = len(batch['input_ids'][0].cpu().tolist())-1

        # Gradients and activations extraction using hooks
        g, a, g_norm, a_norm, g_hist_norm, a_hist_norm, g_distr, a_distr = hooks.get_data(last_token_idx=last_token_idx,
                                                            transformations=transformations,
                                                            norm_type=norm_type,
                                                            stats=True)

        feat_dict = {
            'fact_id': batch['fact_id'][0],
            'class_id': batch['class_id'][0].item(),
            'activations': a,
            'gradients': g,
            'activations_norm': a_norm,
            'gradients_norm': g_norm,
            'activations_hist_norm': a_hist_norm,
            'gradients_hist_norm': g_hist_norm
        }
        

        if save_stats:
            if feat_dict['class_id'] == 0:
                grad_distr[0]=np.concatenate((grad_distr[0],g_distr))
                act_distr[0]=np.concatenate((act_distr[0],a_distr))
                
            else:
                grad_distr[1]=np.concatenate((grad_distr[1],g_distr))
                act_distr[1]=np.concatenate((act_distr[1],a_distr))

        feat_data.append(feat_dict)

        global_step += 1  # Increment global step
        progress_bar.update(1)

    # File saving
    with open(output_path, 'wb') as file:
        pickle.dump(feat_data, file)
    
    print(f"FEATURES EXTRACTION COMPLETED: {len(feat_data)} rows have been generated")

    return grad_distr, act_distr
    

def concatenate_pickles(path_familiar, path_unknown, output_path):
    # Load the first list of dictionaries
    with open(path_familiar, 'rb') as file:
        list1 = pickle.load(file)
    
    # Load the second list of dictionaries
    with open(path_unknown, 'rb') as file:
        list2 = pickle.load(file)
    
    # Concatenate the two lists
    concatenated_list = list1 + list2
    
    # Save the concatenated list to a new pickle file
    with open(output_path, 'wb') as file:
        pickle.dump(concatenated_list, file)
        
    print(f"Concatenated list saved to {output_path}")


def load_json(file_path):
    """
    Load the dataset from a JSON file.
    """
    with open(file_path) as file:
        return json.load(file) 
    


def preprocess_data(entry):
    """
    Preprocess a single data entry to filter and organize the required information.
    """
    paraphrased_facts = list(set(entry['generation_prompts'][1:]).union(entry['paraphrase_prompts']))
    paraphrased_facts = [fact for fact in paraphrased_facts if fact != entry['generation_prompts'][0]]

    return {
        'fact_id': entry['case_id'],
        'prompt': entry['requested_rewrite']['prompt'],
        'target_old': entry['requested_rewrite']['target_true']['str'],
        'target_new': entry['requested_rewrite']['target_new']['str'],
        'important_question': entry['generation_prompts'][0],
        'affiliated_same_true': entry['neighborhood_prompts'][:5],
        'affiliated_same_false': entry['attribute_prompts'][:5],
        'paraphrased_facts': paraphrased_facts
    }



def expand_entry_extraction(entry, unknwon_dataset=False):
    """
    Expand a preprocessed data entry into a format suitable for features
    """
    data_list = []
    if not unknwon_dataset:
    
        data_list.append(
            {'fact_id':str(entry['fact_id'])+'T',
            'question': entry['important_question'],
            'answer': entry['target_old'],
            'answer_correct': entry['target_old'],
            'class_name':"True",
            'class_id':0}
            )
        
        data_list.append(
            {'fact_id':str(entry['fact_id'])+'F',
            'question': entry['important_question'],
            'answer': entry['target_new'],
            'answer_correct': entry['target_old'],
            'class_name':"False",
            'class_id':1}
            )
    else:
        data_list.append(
            {'fact_id':str(entry['fact_id'])+'U',
            'fact': entry['transformed'],
            'class_name':"Unfamiliar",
            'class_id':2}
            )

    return data_list


def tokenize_data_extraction(data_list, tokenizer, unknwon_dataset=False):

    """
    Tokenize the data using the provided tokenizer.
    """
    tokenized_data = []

    if not unknwon_dataset:

        for data in data_list:
            target_text = data['answer']

            tokenizer.pad_token = tokenizer.eos_token
            input_text = f"{data['question']}"
            combined_text = f"{input_text} {target_text}"
            
            tokenized_inputs = tokenizer.encode_plus(combined_text,add_special_tokens=True, truncation=True, return_tensors='pt')
            tokenized_question = tokenizer.encode_plus(input_text,add_special_tokens=True, truncation=True, return_tensors='pt')
            
            input_ids = tokenized_inputs['input_ids'].squeeze(0)
            question_ids = tokenized_question['input_ids'].squeeze(0)

            attention_mask = tokenized_inputs['attention_mask'].squeeze(0)

            question_len = len(question_ids.tolist())
            input_len = len(input_ids.tolist())

            answer_tokens_idx = list(range(question_len, input_len))
            labels = input_ids.clone()


            tokenized_data.append({
                'fact_id': data['fact_id'],
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'answer_tokens_idx': answer_tokens_idx,
                'class_name': data['class_name'],
                'class_id': data['class_id'],
                'text': combined_text,
                'input_text': input_text,
                'target_text': data['answer_correct']
            })
        
    else:

        for data in data_list:
            tokenizer.pad_token = tokenizer.eos_token

            combined_text = data["fact"]

            tokenized_inputs = tokenizer.encode_plus(combined_text,add_special_tokens=True, truncation=True, return_tensors='pt')

            input_ids = tokenized_inputs['input_ids'].squeeze(0)

            attention_mask = tokenized_inputs['attention_mask'].squeeze(0)

            input_len = len(input_ids.tolist())

            answer_tokens_idx = list(range(input_len, input_len))

            labels = input_ids.clone()

            tokenized_data.append({
                'fact_id': data['fact_id'],
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'answer_tokens_idx': answer_tokens_idx,
                'class_name': data['class_name'],
                'class_id': data['class_id'],
                'text': combined_text
            })

    return tokenized_data



def classification_results(X, y, seed, classifier):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Standardize the features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # CV SVM classifier
    if classifier == 'svm':
        # Define parameter grid for the SVM
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 1, 0.1, 0.01, 0.001, ],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],  # Only relevant for the 'poly' kernel
            'coef0': [0.0, 0.1, 0.5, 1.0]  # Used in 'poly' and 'sigmoid' kernels
        }

        estimator = SVC()

    
    # CV RandomForest classifier
    else: 
        # Define the parametergrid for the RF
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        estimator=RandomForestClassifier()

        
        
    with parallel_backend('threading'):
        # Initialize the GridSearchCV object with the parameter grid
        grid_search = GridSearchCV(estimator=estimator,
                                        param_grid=param_grid,
                                        cv=5,
                                        verbose=1,
                                        refit=True,
                                        n_jobs=64)

        # Perform the grid search on the data
        grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Predict the labels for the test set
    y_pred = grid_search.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    print(f'CV+TEST DONE - ACC:{accuracy}')

    return accuracy, f1, report, best_params