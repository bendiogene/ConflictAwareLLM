import os
import json
import random
from tqdm import tqdm
from utils import *
from config_file import *


def load_dataset(file_path):
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
        'target_old': entry['requested_rewrite']['target_true']['str'],
        'target_new': entry['requested_rewrite']['target_new']['str'],
        'important_question': entry['generation_prompts'][0],
        'affiliated_same_true': entry['neighborhood_prompts'][:5],
        'affiliated_same_false': entry['attribute_prompts'][:5],
        'paraphrased_facts': paraphrased_facts
    }

def expand_entry(entry):
    """
    Expand a preprocessed data entry into a format suitable for training.
    """
    data_list = [{'question': entry['important_question'], 'answer': entry['target_old']}]
    for question in entry['affiliated_same_true'] + entry['affiliated_same_false'] + entry['paraphrased_facts']:
        answer = entry['target_old'] if question in entry['affiliated_same_true'] else entry['target_new']
        data_list.append({'question': question, 'answer': answer})
    return data_list

# Load and preprocess dataset
dataset = load_dataset(CONFIG["dataset_file"])
preprocessed_dataset = [preprocess_data(entry) for entry in dataset]
sample_dataset = preprocessed_dataset[:CONFIG["sample_size"]] #Took only the first ones

with open('questions.txt', 'w') as file:
    for entry in sample_dataset:
        # Write each question to the file, followed by a newline character
        file.write(f"{entry['important_question']} {entry['target_old']}\n")

