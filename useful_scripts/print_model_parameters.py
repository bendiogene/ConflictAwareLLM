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
import pandas as pd
warnings.filterwarnings("ignore")

os.environ['CURL_CA_BUNDLE'] = ''


'''
EXPERIMENT 3.1 SCRIPT

'''    

def main():

        model = AutoModelForCausalLM.from_pretrained("models/pt_models/gpt2-small")

        for name, param in model.named_parameters():
              print(name)
              print(param.shape)

if __name__ == "__main__":
    main()


