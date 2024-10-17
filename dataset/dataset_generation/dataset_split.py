import sys
# custom path insertion
sys.path.insert(1, '.')
#from config_file import *
from utils import *
import pickle
from sklearn.model_selection import train_test_split
import argparse
import yaml
import os


"""
Splits the dataset into training and testing sets.

This script is useful for preparing data for neural network training. It makes use of sklearn's train_test_split
function to divide the dataset into training and testing subsets based on the specified test size.

"""
#Load config file
# Parse config file
parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
parser.add_argument('--config', type=str, required=True, help='Path to the Yaml configuration file.')
args = parser.parse_args()
config_path = args.config
config = load_configuration(config_path)
# FT_CONFIG  = config['FT_CONFIG']
# DEVICE_CONFIG = config['DEVICE_CONFIG']
EXTRACTION_CONFIG = config['EXTRACTION_CONFIG']
# LOCATION_CONFIG = config['LOCATION_CONFIG']
# HISTORICAL_FEATURES_CONFIG = config['HISTORICAL_FEATURES_CONFIG']


# Load the dataset from the pickle file
with open(EXTRACTION_CONFIG["save_dataset_path_avg_full"], 'rb') as f:
    dataset = pickle.load(f)

# Split the dataset into train and test sets
train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)

# Save the train set to a new pickle file
with open(f'{EXTRACTION_CONFIG["save_dataset_path_avg_full"].split(".")[0]}_train.pkl', 'wb') as f:
    pickle.dump(train_set, f)

# Save the test set to a new pickle file
with open(f'{EXTRACTION_CONFIG["save_dataset_path_avg_full"].split(".")[0]}_test.pkl', 'wb') as f:
    pickle.dump(test_set, f)

print("Train and test splits generated")