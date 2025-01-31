import sys
# custom path insertion
sys.path.insert(1, '.')

import re
import torch
import numpy as np
from torch.nn.functional import normalize
from utils import *
import pickle
from sklearn.preprocessing import normalize


class ParametersHooks:
    """
    Class which manages the hooks for extracting parameters from the model.

    Args:
        model (torch.nn.Module): HuggingFace Transformers Pytorch language model.
        patterns (List[str]): List of regex patterns used for selecting specific layers (each pattern must identify a single layer in the block).
        history (bool): Specifies whether to use historical features or not.

    Attributes:
        activations (Dict[str, torch.Tensor]): Dictionary with activations of each layer, dimensionality of each value is 
            n_batch x n_tokens x out_dim (first dimension is not considered since we have batch size 1 and we squeeze).
        gradients (Dict[str, torch.Tensor]): Dictionary with output gradients of each layer, dimensionality of each value is 
            n_batch x n_tokens x out_dim (first dimension is not considered since we have batch size 1 and we squeeze).
    """

    def __init__(self, model, patterns, last_token=False, history_path=None, token_dim=True, layer_norm_hist=False):
        self.model = model
        self.patterns = patterns
        self.unified_pattern = '|'.join(patterns)
        self.handles = []  # Store hook handles for cleanup
        self.activations = {}
        self.gradients = {}
        self.layers_idx = [[] for _ in range(len(patterns))]
        self.activations_historical = {}
        self.gradients_historical = {}
        self.history_path = history_path
        self.last_token = last_token
        
        if history_path is not None:
            with open(history_path, 'rb') as file:
                hist_data = pickle.load(file)
                self.grad_hist = hist_data['g_normalization_data']
                self.act_hist = hist_data['a_normalization_data']
        
        self.token_dim = token_dim
        self.layer_norm_hist = layer_norm_hist

    def _attach_hooks(self, historical=False):
        """Shared logic for hook attachment with layer indexing"""
        self.handles = []
        regex_list = [re.compile(p) for p in self.patterns]
        regex = re.compile(self.unified_pattern)
        idx = 0  # Maintain layer index counter

        for name, module in self.model.named_modules():
            if regex.search(name):
                # Match original hook attachment logic
                if historical:
                    fwd_hook = self.forward_hook_historical(name)
                    bwd_hook = self.backward_hook_historical(name)
                else:
                    fwd_hook = self.forward_hook(name)
                    bwd_hook = self.backward_hook(name)

                # Preserve original layer indexing logic
                for i, r in enumerate(regex_list):
                    if r.search(name):
                        self.layers_idx[i].append(idx)
                        break
                idx += 1

                # Register hooks
                self.handles.extend([
                    module.register_forward_hook(fwd_hook),
                    module.register_full_backward_hook(bwd_hook)
                ])
    # Define forward hook -> activations
    def forward_hook(self,layer_name):
        '''
        Activations extraction
        Each time a layer is used we save its activations in a dictionary
        '''
        def hook(module, input, output):
            self.activations[layer_name] = output.detach().squeeze()
        return hook
    
    
    # Define forward hook -> gradients
    def backward_hook(self,layer_name):
        '''
        Gradients extraction
        Each time a layer is used we save its activations in a dictionary
        '''
        def hook(module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0].detach().squeeze()
            
        return hook
    
    
    # Define historical forward hook -> activations
    def forward_hook_historical(self,layer_name):

        '''
        Historical activations extarction
        For the historical extraction we run the code during fine-tuning. It means the instead of having 2D tensors we will have 3D tensors of dimensions
        BATCH_SIZE x N_TOKENS x N_OUTPUT (in normal extraction we have batch size 1 and we perform squeeze).
        In order to extract the historical information we normalize on dimensions (1,2) which means that we normalize each layer output using all layer outputs
        from the given batch. Then we sum the normalized data from the different batches in order to have a 2D tensor as result.
        '''
       
        def hook(module, input, output):
            if self.model.training:

                a=output.detach()
                if self.layer_norm_hist:
                    a_mean, a_std = a.mean(dim=(0, 1, 2), keepdim=True), a.std(dim=(1, 2), keepdim=True) + 1e-6
                    a = (a - a_mean) / a_std

                a=torch.sum(a, dim=0)
                if not self.token_dim:
                    a=torch.sum(a, dim=0)

                if layer_name not in self.activations_historical:
                    self.activations_historical[layer_name] = a.to("cpu")
                else:
                    self.activations_historical[layer_name] += a.to("cpu")
        return hook
        
    
    # Define historical backward hook -> gradients
    def backward_hook_historical(self,layer_name):

        '''
        Historical gradients extraction
        For the historical extraction we run the code during fine-tuning. It means the instead of having 2D tensors we will have 3D tensors of dimensions
        BATCH_SIZE x N_TOKENS x N_OUTPUT (in normal extraction we have batch size 1 and we perform squeeze).
        In order to extract the historical information we normalize on dimensions (1,2) which means that we normalize each layer output using all layer outputs
        from the given batch. Then we sum the normalized data from the different batches in order to have a 2D tensor as result.
        '''
       
        def hook(module, grad_input, grad_output):
            if self.model.training:

                g=grad_output[0].detach()
                
                if self.layer_norm_hist:
                    g_mean, g_std = g.mean(dim=(0, 1, 2), keepdim=True), g.std(dim=(1, 2), keepdim=True) + 1e-6
                    g = (g - g_mean) / g_std
                    
                g=torch.sum(g, dim=0)
                if not self.token_dim:
                    g=torch.sum(g, dim=0)
                
                if layer_name not in self.gradients_historical:
                    self.gradients_historical[layer_name] = g.to("cpu")
                else:
                    self.gradients_historical[layer_name] += g.to("cpu")
        return hook

    # Attach hooks to a model
    def attach_hooks(self):
        """Original non-historical hook attachment"""
        self._attach_hooks(historical=False)

    # Attach historical hooks to a model
    def attach_hooks_historical(self):
        """Historical hook attachment"""
        self._attach_hooks(historical=True)

    def detach_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
        # Maintain original data clearing pattern
        self.activations.clear()
        self.gradients.clear()
        self.activations_historical.clear()
        self.gradients_historical.clear()    

    # Extract data from hooks
    def get_data(self, last_token_idx, transformations, norm_type=None, norm_dim=None, stats=False):
        """
        Retrieves the data collected for a given factor. Data are taken from the dictionary layer by layer and they are normalized.
        Each layer is composed of a 2D array n_tokens x n_dim. Different normalization options are available:
        
        - l2: l2 normalization.
        - std: standard normalization.
        - minmax: min-max scaling.
        
        Args:
            last_token_idx (int): Position of the last token of the sentence (before the padding tokens).
            transformations (list): The ls f rafrmaions to apply. Options are 'all', 'avg', 'std'.
            norm_type (str, optional): The type of normalization to apply. Options are 'l2', 'l2_0', 'std', 'std_0', 'minmax'.
            norm_dim (int, optional): The dimension to normalize.
            history (bool, optional): Whether to apply historical normalization.
            stats (bool, optional): Whether to keep track of distribution data.
        
        Returns:
            list: A list of Numpy array each one referring to a given type of layer. The dimension is N_BLOCKS x OUT_DIM.
        """

        grad_values = [self.gradients[layer].cpu().float().numpy() for layer in self.activations]
        act_values = [self.activations[layer].cpu().float().numpy() for layer in self.activations]

        gradients = []
        activations = []

        # Group the different layers per type. We obtain a list with len equal to the different types of blocks mapped.
        # Each element of the list is a list itself with len euqla to the number of tranformer blocks
        for idx_list in self.layers_idx:
            gradients.append([grad_values[idx] for idx in idx_list])
            activations.append([act_values[idx] for idx in idx_list])

        grad = {}
        act = {}

        grad_norm = {}
        act_norm = {}

        grad_hist_norm = {}
        act_hist_norm = {}

        for t in transformations:
            grad[t] = []
            act[t] = []
            grad_norm[t] = []
            act_norm[t] = []
            grad_hist_norm[t] = []
            act_hist_norm[t] = []


        g_distr, a_distr = (np.array([]), np.array([])) if stats else (None, None)

        for i in range(len(self.patterns)):

            # transform list in a numpy array for parallel operations (N_LAYER x N_TOKENS x OUT_DIM)
            g=np.stack(gradients[i])
            a=np.stack(activations[i])

            if stats:
                g_distr=np.concatenate((g_distr, g[:,last_token_idx-1,:].flatten()))
                a_distr=np.concatenate((a_distr, a[:,last_token_idx,:].flatten()))

            if self.last_token:
                # if we select just the last token we delete the central dimension
                # in this case we have a matrix with dim N_BLOCKS X OUT_DIM
                g_mat = g[:,last_token_idx-1,:].squeeze()
                a_mat = a[:,last_token_idx,:].squeeze()
            else:
                g_mat = g[:,:,:]
                a_mat = a[:,:,:]
            
            # this tuple include all the exis we want to perform the aggregations on. It is all but the first since we want to keep the layer dimension
            axes_agg = tuple(range(1, len(g_mat.shape)))

            for t in transformations:
                if t == 'avg':
                    grad[t].append(np.mean(g_mat, axis=axes_agg))
                    act[t].append(np.mean(a_mat, axis=axes_agg))
                
                elif t == 'std':
                    grad[t].append(np.std(g_mat, axis=axes_agg))
                    act[t].append(np.std(a_mat, axis=axes_agg))
                    
                elif t == 'perc':
                    #print(np.percentile(g_mat, [0, 25, 50, 75, 100], axis=axes_agg).shape)
                    grad[t].append(np.percentile(g_mat, [0, 25, 50, 75, 100], axis=axes_agg))
                    act[t].append(np.percentile(a_mat,  [0, 25, 50, 75, 100], axis=axes_agg))

                elif t == 'all':
                    grad[t].append(g_mat)
                    act[t].append(a_mat)

            # normalization phase (there is a +1 in the dimensions in order to take in account the additional layer dim)
            if norm_type is not None:
                g_norm, a_norm = self._apply_normalization(g, a, norm_type, norm_dim)

                if self.last_token:
                    g_mat = g_norm[:,last_token_idx-1,:].squeeze()
                    a_mat = a_norm[:,last_token_idx,:].squeeze()
                else:
                    g_mat = g_norm
                    a_mat = a_norm
                
                axes_agg = tuple(range(1, len(g_mat.shape)))

                for t in transformations:
                    if t == 'avg':
                        grad_norm[t].append(np.mean(g_mat, axis=axes_agg))
                        act_norm[t].append(np.mean(a_mat, axis=axes_agg))
                    
                    elif t == 'std':
                        grad_norm[t].append(np.std(g_mat, axis=axes_agg))
                        act_norm[t].append(np.std(a_mat, axis=axes_agg))
                        
                    elif t == 'perc':
                        grad_norm[t].append(np.percentile(g_mat, [0, 25, 50, 75, 100], axis=axes_agg))
                        act_norm[t].append(np.percentile(a_mat,  [0, 25, 50, 75, 100], axis=axes_agg))

                    elif t == 'all':
                        grad_norm[t].append(g_mat)
                        act_norm[t].append(a_mat)

            # normalization phase (there is a +1 in the dimensions in order to take in account the additional layer dim)
            if self.history_path is not None:
                g_norm, a_norm = self._apply_normalization(g, a, norm_type, norm_dim)
                g_hist_norm = g_norm / np.array(self.grad_hist[i])[:,np.newaxis,:]
                a_hist_norm = a_norm / np.array(self.act_hist[i])[:,np.newaxis,:]

                if self.last_token:
                    g_mat = g_hist_norm[:,last_token_idx-1,:].squeeze()
                    a_mat = a_hist_norm[:,last_token_idx,:].squeeze()
                else:
                    g_mat = g_hist_norm
                    a_mat = a_hist_norm
                
                axes_agg = tuple(range(1, len(g_mat.shape)))

                for t in transformations:
                    if t == 'avg':
                        grad_hist_norm[t].append(np.mean(g_mat, axis=axes_agg))
                        act_hist_norm[t].append(np.mean(a_mat, axis=axes_agg))
                    
                    elif t == 'std':
                        grad_hist_norm[t].append(np.std(g_mat, axis=axes_agg))
                        act_hist_norm[t].append(np.std(a_mat, axis=axes_agg))
                        
                    elif t == 'perc':
                        grad_hist_norm[t].append(np.percentile(g_mat, [0, 25, 50, 75, 100], axis=axes_agg))
                        act_hist_norm[t].append(np.percentile(a_mat,  [0, 25, 50, 75, 100], axis=axes_agg))

                    elif t == 'all':
                        grad_hist_norm[t].append(g_mat)
                        act_hist_norm[t].append(a_mat)

                   

        for t in transformations:
            grad[t]=np.array(grad[t]).flatten()
            act[t]=np.array(act[t]).flatten()

            grad_norm[t]=np.array(grad_norm[t]).flatten()
            act_norm[t]=np.array(act_norm[t]).flatten()

            grad_hist_norm[t]=np.array(grad_hist_norm[t]).flatten()
            act_hist_norm[t]=np.array(act_hist_norm[t]).flatten()
    
    
        return grad, act, grad_norm, act_norm, grad_hist_norm, act_hist_norm, g_distr, a_distr



    def get_historical_data(self):
        """
        Retrieves historical data for a given fact. Data are taken from the dictionary, layer by layer, and normalized.
        Each layer consists of a 2D array n_tokens x n_dim. Standard deviation normalization is applied in both dimensions.
        The absolute value is computed for gradients only, focusing on the magnitude.
        To better manage the data, each pair of layers is concatenated (e.g., attn_1+attn_2 and mlp_1+mlp_2).

        Returns:
            numpy.ndarray: An array with dimensions (n_layers//2 x n_tokens x out_dim1+out_dim2) containing the processed historical data.
        """
        g_dict = {}
        a_dict = {}

        layers_list = list(self.activations_historical.keys())

        g_dict = {'raw':{}, 'norm':{}}
        a_dict = {'raw':{}, 'norm':{}}

        for layer in layers_list:
                    
            g = self.gradients_historical[layer]
            a = self.activations_historical[layer]

            # bf16 does not work with hooks
            #g_dict['raw'][layer] = g.cpu().numpy()
            g_dict['raw'][layer] = g.cpu().float().numpy() #zi: to check
            #a_dict['raw'][layer] = a.cpu().numpy()
            a_dict['raw'][layer] = a.cpu().float().numpy() # zi: to check for bf16 compatibilitys

            g_mean, g_std = g.mean(), g.std() + 1e-6
            g_norm = (g - g_mean) / g_std

            a_mean, a_std = a.mean(), a.std() + 1e-6
            a_norm = (a - a_mean) / a_std

            
            g_dict['norm'][layer] = torch.abs(g_norm).cpu().float().numpy()
            a_dict['norm'][layer] = torch.abs(a_norm).cpu().float().numpy()

        grad_values = [g_dict['norm'][layer] for layer in layers_list]
        act_values = [a_dict['norm'][layer] for layer in layers_list]

        g_normalization_data = []
        a_normalization_data = []

        for idx_list in self.layers_idx:
            g_normalization_data.append([grad_values[idx] for idx in idx_list])
            a_normalization_data.append([act_values[idx] for idx in idx_list])

        return g_dict, a_dict, g_normalization_data,  a_normalization_data
    

    
    def get_historical_data_sum(self):
        """
        Retrieves historical data for a given fact. Data are taken from the dictionary, layer by layer, and normalized.
        Each layer consists of a 2D array n_tokens x n_dim. Standard deviation normalization is applied in both dimensions.
        The absolute value is computed for gradients only, focusing on the magnitude.
        To better manage the data, each pair of layers is concatenated (e.g., attn_1+attn_2 and mlp_1+mlp_2).

        Returns:
            numpy.ndarray: An array with dimensions (n_layers//2 x n_tokens x out_dim1+out_dim2) containing the processed historical data.
        """
        g_dict = {}
        a_dict = {}

        layers_list = list(self.activations_historical.keys())

        g_dict = {'raw':{}, 'norm':{}}
        a_dict = {'raw':{}, 'norm':{}}

        
        for layer in layers_list:
                    
            g = self.gradients_historical[layer]
            a = self.activations_historical[layer]

            sum_g=torch.sum(g, axis=0)
            sum_a=torch.sum(a, axis=0)

            g_dict['raw'][layer] = sum_g.cpu().float().numpy()
            a_dict['raw'][layer] = sum_a.cpu().float().numpy()
            
            g_mean, g_std = sum_g.mean(), sum_g.std() + 1e-6
            sum_g_norm = (sum_g - g_mean) / g_std

            a_mean, a_std = sum_a.mean(), sum_a.std() + 1e-6
            sum_a_norm = (sum_a - a_mean) / a_std 

            g_dict['norm'][layer] = sum_g_norm.cpu().float().numpy()
            a_dict['norm'][layer] = sum_a_norm.cpu().float().numpy()

        return g_dict, a_dict

    
    def get_top_gradients(self,k,idx):
        """
        Retrieves the indexes of the top k output gradients for each layer at a specific final token index.
        
        Args:
            k (int): The number of gradients to retrieve.
            idx (int): The final token index to consider for gradients.
        
        Returns:
            numpy.ndarray: An array of shape (n_layers x k) containing the indexes of the top k gradients for each layer.
        """
        top_grad = []

        layers_list = list(self.activations.keys())
        
        for i,layer in enumerate(layers_list):
            #zi: is this correct? to check
            idxs=np.argsort(np.abs(self.gradients[layer][idx].cpu().float().numpy()))[-k:]
            top_grad.append(idxs)
        
        return np.array(top_grad)
    

    
    def _apply_normalization(self, g, a, norm_type, norm_dim):
        """
        Applies normalization.
        
        Args:
            g (numpy.ndarray): Gradients values.
            a (numpy.ndarray): Activations values.
            norm_type (str): The type of normalization to apply. Options are 'l2', 'l2_0', 'std', 'std_0', 'minmax'.
            norm_dim (int): The dimension to normalize.
        
        Returns:
            numpy.ndarray: An array of shape (n_layers x k) containing the indexes of the top k gradients for each layer.
        """
        if norm_dim is None:
            norm_dim = (1,2)
            a_dim = (a.shape[0],-1)
            g_dim = (g.shape[0],-1)
        else:
            a_dim = a.shape
            g_dim = g.shape

        if norm_type == 'l2':
            g = normalize(g.reshape(g_dim), norm='l2').reshape(g.shape)
            a = normalize(a.reshape(a_dim), norm='l2').reshape(a.shape)

        elif norm_type == 'std':
            g_mean, g_std = g.mean(axis=norm_dim, keepdims=True), g.std(axis=(1, 2), keepdims=True) + 1e-6
            a_mean, a_std = a.mean(axis=norm_dim, keepdims=True), a.std(axis=(1, 2), keepdims=True) + 1e-6
            g = (g - g_mean) / g_std
            a = (a - a_mean) / a_std

        elif norm_type == 'minmax':
            g_min, g_max = g.min(axis=norm_dim, keepdims=True), g.max(axis=(1, 2), keepdims=True)
            a_min, a_max = a.min(axis=norm_dim, keepdims=True), a.max(axis=(1, 2), keepdims=True)
            g = (g - g_min) / (g_max - g_min + 1e-6)
            a = (a - a_min) / (a_max - a_min + 1e-6)

        return g, a
        
# import sys
# # custom path insertion
# sys.path.insert(1, '.')

# import re
# import torch
# import numpy as np
# from torch.nn.functional import normalize
# from utils import *
# import pickle
# from sklearn.preprocessing import normalize


# class ParametersHooks:

#     """
#     Class which manages the hooks for extracting parameters from the model.

#     Args:
#         model (torch.nn.Module): HuggingFace Transformers Pytorch language model.
#         patterns (List[str]): List of regex patterns used for selecting specific layers (each pattern must identify a single layer in the block).
#         history (bool): Specifies whether to use historical features or not.

#     Attributes:
#         activations (Dict[str, torch.Tensor]): Dictionary with activations of each layer, dimensionality of each value is 
#             n_batch x n_tokens x out_dim (first dimension is not considered since we have batch size 1 and we squeeze).
#         gradients (Dict[str, torch.Tensor]): Dictionary with output gradients of each layer, dimensionality of each value is 
#             n_batch x n_tokens x out_dim (first dimension is not considered since we have batch size 1 and we squeeze).
#     """

#     def __init__(self, model, patterns, last_token=False, history_path=None, token_dim=True, layer_norm_hist=False):
#         self.model = model
#         self.patterns = patterns
#         self.unified_pattern = '|'.join(patterns)
#         self.activations = {}
#         self.gradients = {}
#         self.layers_idx = [[] for _ in range(len(patterns))]
#         self.activations_historical = {}
#         self.gradients_historical = {}
#         self.history_path = history_path
#         self.last_token = last_token
        
#         if history_path is not None:
#             with open(history_path, 'rb') as file:
#                 hist_data = pickle.load(file)
#                 self.grad_hist = hist_data['g_normalization_data']
#                 self.act_hist = hist_data['a_normalization_data']
        
#         self.token_dim = token_dim
#         self.layer_norm_hist = layer_norm_hist


#     # Define forward hook -> activations
#     def forward_hook(self,layer_name):
#         '''
#         Activations extraction
#         Each time a layer is used we save its activations in a dictionary
#         '''
#         def hook(module, input, output):
#             self.activations[layer_name] = output.detach().squeeze()
#         return hook
    
    
#     # Define forward hook -> gradients
#     def backward_hook(self,layer_name):
#         '''
#         Gradients extraction
#         Each time a layer is used we save its activations in a dictionary
#         '''
#         def hook(module, grad_input, grad_output):
#             self.gradients[layer_name] = grad_output[0].detach().squeeze()
            
#         return hook
    
    
#     # Define historical forward hook -> activations
#     def forward_hook_historical(self,layer_name):

#         '''
#         Historical activations extarction
#         For the historical extraction we run the code during fine-tuning. It means the instead of having 2D tensors we will have 3D tensors of dimensions
#         BATCH_SIZE x N_TOKENS x N_OUTPUT (in normal extraction we have batch size 1 and we perform squeeze).
#         In order to extract the historical information we normalize on dimensions (1,2) which means that we normalize each layer output using all layer outputs
#         from the given batch. Then we sum the normalized data from the different batches in order to have a 2D tensor as result.
#         '''
       
#         def hook(module, input, output):
#             if self.model.training:

#                 a=output.detach()
#                 if self.layer_norm_hist:
#                     a_mean, a_std = a.mean(dim=(0, 1, 2), keepdim=True), a.std(dim=(1, 2), keepdim=True) + 1e-6
#                     a = (a - a_mean) / a_std

#                 a=torch.sum(a, dim=0)
#                 if not self.token_dim:
#                     a=torch.sum(a, dim=0)

#                 if layer_name not in self.activations_historical:
#                     self.activations_historical[layer_name] = a.to("cpu")
#                 else:
#                     self.activations_historical[layer_name] += a.to("cpu")
#         return hook
        
    
#     # Define historical backward hook -> gradients
#     def backward_hook_historical(self,layer_name):

#         '''
#         Historical gradients extraction
#         For the historical extraction we run the code during fine-tuning. It means the instead of having 2D tensors we will have 3D tensors of dimensions
#         BATCH_SIZE x N_TOKENS x N_OUTPUT (in normal extraction we have batch size 1 and we perform squeeze).
#         In order to extract the historical information we normalize on dimensions (1,2) which means that we normalize each layer output using all layer outputs
#         from the given batch. Then we sum the normalized data from the different batches in order to have a 2D tensor as result.
#         '''
       
#         def hook(module, grad_input, grad_output):
#             if self.model.training:

#                 g=grad_output[0].detach()
                
#                 if self.layer_norm_hist:
#                     g_mean, g_std = g.mean(dim=(0, 1, 2), keepdim=True), g.std(dim=(1, 2), keepdim=True) + 1e-6
#                     g = (g - g_mean) / g_std
                    
#                 g=torch.sum(g, dim=0)
#                 if not self.token_dim:
#                     g=torch.sum(g, dim=0)
                
#                 if layer_name not in self.gradients_historical:
#                     self.gradients_historical[layer_name] = g.to("cpu")
#                 else:
#                     self.gradients_historical[layer_name] += g.to("cpu")
#         return hook


#     # Attach hooks to a model
#     def attach_hooks(self): # Add other layer types as needed
#         '''
#         Attach hooks
#         This function perform the actual hooks connection
#         '''
#         idx = 0
#         regex_list = [re.compile(pattern) for pattern in self.patterns]
#         regex = re.compile(self.unified_pattern)
#         for name, module in self.model.named_modules():
#             if regex.search(name):
#                 # Correctly capture `name` for each layer by passing it to the hook functions
#                 module.register_forward_hook(self.forward_hook(name))
#                 module.register_full_backward_hook(self.backward_hook(name))

#                 for i,r in enumerate(regex_list):
#                     if r.search(name):
#                         self.layers_idx[i].append(idx)
#                         break
#                 idx+=1


#     # Attach historical hooks to a model
#     def attach_hooks_historical(self): # Add other layer types as needed
#         '''
#         Attach hooks hisotircal
#         This function perform the actual hooks connection for historical data
#         '''
#         idx = 0
#         regex_list = [re.compile(pattern) for pattern in self.patterns]
#         regex = re.compile(self.unified_pattern)
#         for name, module in self.model.named_modules():
#             if regex.search(name):
#                 # Correctly capture `name` for each layer by passing it to the hook functions
#                 module.register_forward_hook(self.forward_hook_historical(name))
#                 module.register_full_backward_hook(self.backward_hook_historical(name))

#                 for i,r in enumerate(regex_list):
#                     if r.search(name):
#                         self.layers_idx[i].append(idx)
#                         break
#                 idx+=1
    

#     # Extract data from hooks
#     def get_data(self, last_token_idx, transformations, norm_type=None, norm_dim=None, stats=False):
#         """
#         Retrieves the data collected for a given factor. Data are taken from the dictionary layer by layer and they are normalized.
#         Each layer is composed of a 2D array n_tokens x n_dim. Different normalization options are available:
        
#         - l2: l2 normalization.
#         - std: standard normalization.
#         - minmax: min-max scaling.
        
#         Args:
#             last_token_idx (int): Position of the last token of the sentence (before the padding tokens).
#             transformations (list): The ls f rafrmaions to apply. Options are 'all', 'avg', 'std'.
#             norm_type (str, optional): The type of normalization to apply. Options are 'l2', 'l2_0', 'std', 'std_0', 'minmax'.
#             norm_dim (int, optional): The dimension to normalize.
#             history (bool, optional): Whether to apply historical normalization.
#             stats (bool, optional): Whether to keep track of distribution data.
        
#         Returns:
#             list: A list of Numpy array each one referring to a given type of layer. The dimension is N_BLOCKS x OUT_DIM.
#         """

#         grad_values = [self.gradients[layer].cpu().float().numpy() for layer in self.activations]
#         act_values = [self.activations[layer].cpu().float().numpy() for layer in self.activations]

#         gradients = []
#         activations = []

#         # Group the different layers per type. We obtain a list with len equal to the different types of blocks mapped.
#         # Each element of the list is a list itself with len euqla to the number of tranformer blocks
#         for idx_list in self.layers_idx:
#             gradients.append([grad_values[idx] for idx in idx_list])
#             activations.append([act_values[idx] for idx in idx_list])

#         grad = {}
#         act = {}

#         grad_norm = {}
#         act_norm = {}

#         grad_hist_norm = {}
#         act_hist_norm = {}

#         for t in transformations:
#             grad[t] = []
#             act[t] = []
#             grad_norm[t] = []
#             act_norm[t] = []
#             grad_hist_norm[t] = []
#             act_hist_norm[t] = []


#         g_distr, a_distr = (np.array([]), np.array([])) if stats else (None, None)

#         for i in range(len(self.patterns)):

#             # transform list in a numpy array for parallel operations (N_LAYER x N_TOKENS x OUT_DIM)
#             g=np.stack(gradients[i])
#             a=np.stack(activations[i])

#             if stats:
#                 g_distr=np.concatenate((g_distr, g[:,last_token_idx-1,:].flatten()))
#                 a_distr=np.concatenate((a_distr, a[:,last_token_idx,:].flatten()))

#             if self.last_token:
#                 # if we select just the last token we delete the central dimension
#                 # in this case we have a matrix with dim N_BLOCKS X OUT_DIM
#                 g_mat = g[:,last_token_idx-1,:].squeeze()
#                 a_mat = a[:,last_token_idx,:].squeeze()
#             else:
#                 g_mat = g[:,:,:]
#                 a_mat = a[:,:,:]
            
#             # this tuple include all the exis we want to perform the aggregations on. It is all but the first since we want to keep the layer dimension
#             axes_agg = tuple(range(1, len(g_mat.shape)))

#             for t in transformations:
#                 if t == 'avg':
#                     grad[t].append(np.mean(g_mat, axis=axes_agg))
#                     act[t].append(np.mean(a_mat, axis=axes_agg))
                
#                 elif t == 'std':
#                     grad[t].append(np.std(g_mat, axis=axes_agg))
#                     act[t].append(np.std(a_mat, axis=axes_agg))
                    
#                 elif t == 'perc':
#                     #print(np.percentile(g_mat, [0, 25, 50, 75, 100], axis=axes_agg).shape)
#                     grad[t].append(np.percentile(g_mat, [0, 25, 50, 75, 100], axis=axes_agg))
#                     act[t].append(np.percentile(a_mat,  [0, 25, 50, 75, 100], axis=axes_agg))

#                 elif t == 'all':
#                     grad[t].append(g_mat)
#                     act[t].append(a_mat)

#             # normalization phase (there is a +1 in the dimensions in order to take in account the additional layer dim)
#             if norm_type is not None:
#                 g_norm, a_norm = self._apply_normalization(g, a, norm_type, norm_dim)

#                 if self.last_token:
#                     g_mat = g_norm[:,last_token_idx-1,:].squeeze()
#                     a_mat = a_norm[:,last_token_idx,:].squeeze()
#                 else:
#                     g_mat = g_norm
#                     a_mat = a_norm
                
#                 axes_agg = tuple(range(1, len(g_mat.shape)))

#                 for t in transformations:
#                     if t == 'avg':
#                         grad_norm[t].append(np.mean(g_mat, axis=axes_agg))
#                         act_norm[t].append(np.mean(a_mat, axis=axes_agg))
                    
#                     elif t == 'std':
#                         grad_norm[t].append(np.std(g_mat, axis=axes_agg))
#                         act_norm[t].append(np.std(a_mat, axis=axes_agg))
                        
#                     elif t == 'perc':
#                         grad_norm[t].append(np.percentile(g_mat, [0, 25, 50, 75, 100], axis=axes_agg))
#                         act_norm[t].append(np.percentile(a_mat,  [0, 25, 50, 75, 100], axis=axes_agg))

#                     elif t == 'all':
#                         grad_norm[t].append(g_mat)
#                         act_norm[t].append(a_mat)

#             # normalization phase (there is a +1 in the dimensions in order to take in account the additional layer dim)
#             if self.history_path is not None:
#                 g_norm, a_norm = self._apply_normalization(g, a, norm_type, norm_dim)
#                 g_hist_norm = g_norm / np.array(self.grad_hist[i])[:,np.newaxis,:]
#                 a_hist_norm = a_norm / np.array(self.act_hist[i])[:,np.newaxis,:]

#                 if self.last_token:
#                     g_mat = g_hist_norm[:,last_token_idx-1,:].squeeze()
#                     a_mat = a_hist_norm[:,last_token_idx,:].squeeze()
#                 else:
#                     g_mat = g_hist_norm
#                     a_mat = a_hist_norm
                
#                 axes_agg = tuple(range(1, len(g_mat.shape)))

#                 for t in transformations:
#                     if t == 'avg':
#                         grad_hist_norm[t].append(np.mean(g_mat, axis=axes_agg))
#                         act_hist_norm[t].append(np.mean(a_mat, axis=axes_agg))
                    
#                     elif t == 'std':
#                         grad_hist_norm[t].append(np.std(g_mat, axis=axes_agg))
#                         act_hist_norm[t].append(np.std(a_mat, axis=axes_agg))
                        
#                     elif t == 'perc':
#                         grad_hist_norm[t].append(np.percentile(g_mat, [0, 25, 50, 75, 100], axis=axes_agg))
#                         act_hist_norm[t].append(np.percentile(a_mat,  [0, 25, 50, 75, 100], axis=axes_agg))

#                     elif t == 'all':
#                         grad_hist_norm[t].append(g_mat)
#                         act_hist_norm[t].append(a_mat)

                   

#         for t in transformations:
#             grad[t]=np.array(grad[t]).flatten()
#             act[t]=np.array(act[t]).flatten()

#             grad_norm[t]=np.array(grad_norm[t]).flatten()
#             act_norm[t]=np.array(act_norm[t]).flatten()

#             grad_hist_norm[t]=np.array(grad_hist_norm[t]).flatten()
#             act_hist_norm[t]=np.array(act_hist_norm[t]).flatten()
    
    
#         return grad, act, grad_norm, act_norm, grad_hist_norm, act_hist_norm, g_distr, a_distr



#     def get_historical_data(self):
#         """
#         Retrieves historical data for a given fact. Data are taken from the dictionary, layer by layer, and normalized.
#         Each layer consists of a 2D array n_tokens x n_dim. Standard deviation normalization is applied in both dimensions.
#         The absolute value is computed for gradients only, focusing on the magnitude.
#         To better manage the data, each pair of layers is concatenated (e.g., attn_1+attn_2 and mlp_1+mlp_2).

#         Returns:
#             numpy.ndarray: An array with dimensions (n_layers//2 x n_tokens x out_dim1+out_dim2) containing the processed historical data.
#         """
#         g_dict = {}
#         a_dict = {}

#         layers_list = list(self.activations_historical.keys())

#         g_dict = {'raw':{}, 'norm':{}}
#         a_dict = {'raw':{}, 'norm':{}}

#         for layer in layers_list:
                    
#             g = self.gradients_historical[layer]
#             a = self.activations_historical[layer]

#             # bf16 does not work with hooks
#             #g_dict['raw'][layer] = g.cpu().numpy()
#             g_dict['raw'][layer] = g.cpu().float().numpy() #zi: to check
#             #a_dict['raw'][layer] = a.cpu().numpy()
#             a_dict['raw'][layer] = a.cpu().float().numpy() # zi: to check for bf16 compatibilitys

#             g_mean, g_std = g.mean(), g.std() + 1e-6
#             g_norm = (g - g_mean) / g_std

#             a_mean, a_std = a.mean(), a.std() + 1e-6
#             a_norm = (a - a_mean) / a_std

            
#             g_dict['norm'][layer] = torch.abs(g_norm).cpu().float().numpy()
#             a_dict['norm'][layer] = torch.abs(a_norm).cpu().float().numpy()

#         grad_values = [g_dict['norm'][layer] for layer in layers_list]
#         act_values = [a_dict['norm'][layer] for layer in layers_list]

#         g_normalization_data = []
#         a_normalization_data = []

#         for idx_list in self.layers_idx:
#             g_normalization_data.append([grad_values[idx] for idx in idx_list])
#             a_normalization_data.append([act_values[idx] for idx in idx_list])

#         return g_dict, a_dict, g_normalization_data,  a_normalization_data
    

    
#     def get_historical_data_sum(self):
#         """
#         Retrieves historical data for a given fact. Data are taken from the dictionary, layer by layer, and normalized.
#         Each layer consists of a 2D array n_tokens x n_dim. Standard deviation normalization is applied in both dimensions.
#         The absolute value is computed for gradients only, focusing on the magnitude.
#         To better manage the data, each pair of layers is concatenated (e.g., attn_1+attn_2 and mlp_1+mlp_2).

#         Returns:
#             numpy.ndarray: An array with dimensions (n_layers//2 x n_tokens x out_dim1+out_dim2) containing the processed historical data.
#         """
#         g_dict = {}
#         a_dict = {}

#         layers_list = list(self.activations_historical.keys())

#         g_dict = {'raw':{}, 'norm':{}}
#         a_dict = {'raw':{}, 'norm':{}}

        
#         for layer in layers_list:
                    
#             g = self.gradients_historical[layer]
#             a = self.activations_historical[layer]

#             sum_g=torch.sum(g, axis=0)
#             sum_a=torch.sum(a, axis=0)

#             g_dict['raw'][layer] = sum_g.cpu().float().numpy()
#             a_dict['raw'][layer] = sum_a.cpu().float().numpy()
            
#             g_mean, g_std = sum_g.mean(), sum_g.std() + 1e-6
#             sum_g_norm = (sum_g - g_mean) / g_std

#             a_mean, a_std = sum_a.mean(), sum_a.std() + 1e-6
#             sum_a_norm = (sum_a - a_mean) / a_std 

#             g_dict['norm'][layer] = sum_g_norm.cpu().float().numpy()
#             a_dict['norm'][layer] = sum_a_norm.cpu().float().numpy()

#         return g_dict, a_dict

    
#     def get_top_gradients(self,k,idx):
#         """
#         Retrieves the indexes of the top k output gradients for each layer at a specific final token index.
        
#         Args:
#             k (int): The number of gradients to retrieve.
#             idx (int): The final token index to consider for gradients.
        
#         Returns:
#             numpy.ndarray: An array of shape (n_layers x k) containing the indexes of the top k gradients for each layer.
#         """
#         top_grad = []

#         layers_list = list(self.activations.keys())
        
#         for i,layer in enumerate(layers_list):
#             #zi: is this correct? to check
#             idxs=np.argsort(np.abs(self.gradients[layer][idx].cpu().float().numpy()))[-k:]
#             top_grad.append(idxs)
        
#         return np.array(top_grad)
    

    
#     def _apply_normalization(self, g, a, norm_type, norm_dim):
#         """
#         Applies normalization.
        
#         Args:
#             g (numpy.ndarray): Gradients values.
#             a (numpy.ndarray): Activations values.
#             norm_type (str): The type of normalization to apply. Options are 'l2', 'l2_0', 'std', 'std_0', 'minmax'.
#             norm_dim (int): The dimension to normalize.
        
#         Returns:
#             numpy.ndarray: An array of shape (n_layers x k) containing the indexes of the top k gradients for each layer.
#         """
#         if norm_dim is None:
#             norm_dim = (1,2)
#             a_dim = (a.shape[0],-1)
#             g_dim = (g.shape[0],-1)
#         else:
#             a_dim = a.shape
#             g_dim = g.shape

#         if norm_type == 'l2':
#             g = normalize(g.reshape(g_dim), norm='l2').reshape(g.shape)
#             a = normalize(a.reshape(a_dim), norm='l2').reshape(a.shape)

#         elif norm_type == 'std':
#             g_mean, g_std = g.mean(axis=norm_dim, keepdims=True), g.std(axis=(1, 2), keepdims=True) + 1e-6
#             a_mean, a_std = a.mean(axis=norm_dim, keepdims=True), a.std(axis=(1, 2), keepdims=True) + 1e-6
#             g = (g - g_mean) / g_std
#             a = (a - a_mean) / a_std

#         elif norm_type == 'minmax':
#             g_min, g_max = g.min(axis=norm_dim, keepdims=True), g.max(axis=(1, 2), keepdims=True)
#             a_min, a_max = a.min(axis=norm_dim, keepdims=True), a.max(axis=(1, 2), keepdims=True)
#             g = (g - g_min) / (g_max - g_min + 1e-6)
#             a = (a - a_min) / (a_max - a_min + 1e-6)

#         return g, a
        