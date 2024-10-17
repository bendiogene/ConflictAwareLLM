import re
import torch
import numpy as np
from torch.nn.functional import normalize
from utils import *


class WeightsTracker:
    
    """
    Class which manages the weights gradient extraction.

    Args:
        model: HuggingFace Transformers Pytorch language model
        pattern: regex expression for selecting specific layers

    Attr:
        gradeints: dictionary with weights gradients for each layer. We have two arrays for layer describing weights and biases:
            dimensionality of each weights array is in_dim x out_dim, while for biases is out_dim
        gradients_historical: dictionary with historical weights gradients for each layer. We have two arrays for layer describing weights and biases:
            dimensionality of each weights array is in_dim x out_dim, while for biases is out_dim
    """
    def __init__(self,model,pattern):
        self.model = model
        self.pattern = pattern
        self.gradients = {}  # To store parameter gradients
        self.gradients_historical = {}


    def extract_gradients(self):
        '''
        Gradients extarction
        Each time it is called we save all mapped layer's parameters gradients in a dictionary.
        Considering we are dealing with model's parameters directly it is not possible to use hooks and this process must be done manually.
        '''
        for name, param in self.model.named_parameters():
            regex = re.compile(self.pattern)
            if regex.search(name):
                if param.grad is not None:
                    self.gradients[name] = param.grad.data.clone().cpu().numpy()



    def get_top_gradients(self,k,idx):
        '''
        Top gradients retrieval
        in:
            k: number of gradients
            idx: final token index
        out:
            numpy array (n_layers x k)
        This function return the indexes of the top k output gradients for each layer.
        '''
        self.extract_gradients()
        top_weights = []
        layers_list = list(self.gradients.keys())

        for i,layer in enumerate(layers_list):
            top_weights.append(np.argsort(np.abs(self.gradients[layer][idx]))[-k:])
    
        return top_weights