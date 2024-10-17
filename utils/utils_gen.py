import os
import json
import random
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, TrainerCallback
import torch.nn.functional as F
import yaml


def expand_entry(entry):
    """
    Expand a preprocessed data entry into a format suitable for training.
    """
    data_list = [{'question': entry['important_question'], 'answer': entry['target_old']}]
    for question in entry['affiliated_same_true'] + entry['affiliated_same_false'] + entry['paraphrased_facts']:
        answer = entry['target_old'] if question in entry['affiliated_same_true'] else entry['target_new']
        data_list.append({'question': question, 'answer': answer})
    return data_list

def format_entry(entry):
    """
    Extract the true fact from the dataset.
    """
    return {'question': entry['important_question'], 'answer': entry['target_old']}


def tokenize_data(data_list, tokenizer, max_length, conditional_generation_mode,  model_name, simple_out=False):

    """
    Tokenize the data using the provided tokenizer.
    """
    tokenized_data = []

    for data in data_list:
        target_text = data['answer']

        tokenizer.pad_token = tokenizer.eos_token
        input_text = f"{data['question']}"
        combined_text = f"{input_text} {target_text}"
        
        tokenized_inputs = tokenizer.encode_plus(combined_text,add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        tokenized_question = tokenizer.encode_plus(input_text,add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        tokenized_answer = tokenizer.encode_plus(target_text,add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        input_ids = tokenized_inputs['input_ids'].squeeze(0)
        question_ids = tokenized_question['input_ids'].squeeze(0)
        answer_ids = tokenized_answer['input_ids'].squeeze(0)

        attention_mask = tokenized_inputs['attention_mask'].squeeze(0)

        question_len = question_ids.tolist().index(tokenizer.eos_token_id)
        input_len = input_ids.tolist().index(tokenizer.eos_token_id)

        answer_tokens_idx = list(range(question_len,input_len))

        # Focus your attention on the answer generation only
        if conditional_generation_mode =="focus_on_answer":
            # Initialize all labels with -100
            labels = torch.full(input_ids.shape, -100)
            # Unmask only the anser tokens (the tokens in input and not in question)
            labels[question_len:] = input_ids[question_len:]

        # Learn to generate the entire sequence, don't mask the labels
        else:
            labels = input_ids.clone()

        if simple_out:
            tokenized_data.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
            })
            
        else:
            tokenized_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'answer_tokens_idx': answer_tokens_idx,
                'question_tokens': question_ids,
                'answer_ids': answer_ids,
                'answer': target_text
                })

    return tokenized_data

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

def tokenize_data_extraction(data_list, tokenizer, max_length, unknwon_dataset=False):

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
            
            tokenized_inputs = tokenizer.encode_plus(combined_text,add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
            tokenized_question = tokenizer.encode_plus(input_text,add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
            
            input_ids = tokenized_inputs['input_ids'].squeeze(0)
            question_ids = tokenized_question['input_ids'].squeeze(0)

            attention_mask = tokenized_inputs['attention_mask'].squeeze(0)

            question_len = question_ids.tolist().index(tokenizer.eos_token_id)
            input_len = input_ids.tolist().index(tokenizer.eos_token_id)

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

            tokenized_inputs = tokenizer.encode_plus(combined_text,add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

            input_ids = tokenized_inputs['input_ids'].squeeze(0)

            attention_mask = tokenized_inputs['attention_mask'].squeeze(0)

            input_len = input_ids.tolist().index(tokenizer.eos_token_id)

            answer_tokens_idx = list(range(input_len-1, input_len))

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

def pad_tensor(tensor, target_dim, pad_value=0):
    padding_needed = target_dim - tensor.size(0)
    if padding_needed > 0:
        # The padding is applied in the form (padLeft, padRight, padTop, padBottom, ...)
        # For a 3D tensor, we want to pad the first dimension (depth/front/back), hence (0,0) for the last two dimensions
        # and padding_needed at the start of the first dimension. No padding is applied to the middle dimension.
        pad = (0, 0, 0, 0, 0, 0)  # Padding format: (W_left, W_right, H_top, H_bottom, D_front, D_back)
        padded_tensor = F.pad(tensor, pad, "constant", pad_value)
    else:
        padded_tensor = tensor
    return padded_tensor

def load_configuration(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)