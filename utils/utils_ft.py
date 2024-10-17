import sys
# custom path insertion

sys.path.insert(1, '.')
import utils
from utils.parameters_hooks import *
from utils.utils_evaluate import *

# from utils import ParametersHooks, verify_acc
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

import shutil
import tempfile
import gc


"""
Fine-tuning function (trainer)
This is a fine-tuning function built with HuggingFace trainer. It incorporates the historical extraction to store historical features during training.
"""
def ft_trainer(model, lr, num_epochs, batch_size, dataset_update, dataset_eval, out_dir, logging_dir, padding_mask, pad_token, patterns=None, historical_path=None, save_model=True, layer_norm = False):

    if historical_path is not None:
        hooks = ParametersHooks(model=model, patterns=patterns, token_dim=False, layer_norm_hist=layer_norm)
        hooks.attach_hooks_historical()

    tmp_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=tmp_dir, 
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_update
    )
    
    trainer.train()

    if historical_path is not None:
        g_dict, a_dict, g_norm, a_norm = hooks.get_historical_data()

        arrays_dict = {
            'act_hist_dict': a_dict, 
            'grad_hist_dict': g_dict,
            'a_normalization_data': a_norm,
            'g_normalization_data': g_norm
        }

        with open(historical_path, 'wb') as file:
            pickle.dump(arrays_dict, file)

    if save_model:
        model.save_pretrained(out_dir)

    shutil.rmtree(tmp_dir)

    score_update = verify_acc(model, dataset_update, padding_mask, pad_token)
    score_eval = verify_acc(model, dataset_eval, padding_mask, pad_token)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return score_update, score_eval

            
"""
Fine-tuning function (custom)
This is a fine-tuning function based on EasyEdit code. It implements the freezing option to block the update on given neurons.
"""    
def ft_custom(
    model, 
    lr,
    num_epochs,
    batch_size,
    dataset_update,
    dataset_eval,
    out_dir, 
    logging_dir,
    padding_mask,
    pad_token,
    dataset_generality = None,
    patterns=None,
    historical_path=None,
    gen_hist_path=None,
    rnd=False,
    norm=False,
    threshold=0,
    freeze=False,
    save_model=True,
    inverse=False,
    ev_epochs=False,
    freeze_other_layers = True
    ):
    
    # Initialize Accelerator
    accelerator = Accelerator() 

    # Extract stubborn neurons (If freeze is true it means that we want to freeze the stubborn neurons)
    if freeze:
        # When the general_hist is not specified the model applies the standard neurons selection
        if gen_hist_path is None:
            stubborn_idx, neurons_per_layer, stubborn_number = stubborn_extraction(historical_path, norm, threshold, inverse)
        # This is the case when we want to apply the intersection based neurons selection 
        else:
            stubborn_idx, neurons_per_layer, stubborn_number = stubborn_extraction_plus(historical_path, gen_hist_path, threshold)

        print(f'Stubborn number: {stubborn_number}')

        # If the random option is selected the model generates a random distribution of neurons with the same cardinality of the stubborns
        if rnd:
            rnd_distr = generate_rnd_distr(stubborn_number, len(stubborn_idx.keys()), neurons_per_layer)
            random_idx = {}

            rnd_number = 0
            for i,elem in enumerate(stubborn_idx): 
                rng = default_rng()
                random_idx[elem]=rng.choice(neurons_per_layer[i], size=rnd_distr[i], replace=False)
                rnd_number += len(random_idx[elem])

            print(f'Random number: {rnd_number}')
            stubborn_idx=random_idx


    # Configure optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Create dataset
    train_dataset = dataset_update.with_format(type='torch')

    # Create dataloader
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # loss definition
    loss_fct = CrossEntropyLoss(reduction='none')

    # Prepare model, optimizer, and dataloader for acceleration
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_epochs=num_epochs
    num_training_steps = len(dataloader) * num_epochs
    
    progress_bar = tqdm(range(num_training_steps))

    # Training loop
    model.train()
    global_step = 0

    epoch_acc_update=[]
    epoch_acc_eval=[]
   
    for epoch in range(num_epochs):
        cum_loss = 0
        model.train()
        for batch in dataloader:
            # define batch size
            bs = batch['input_ids'].shape[0]

            # define input variables
            inputs_targets={'input_ids': batch['input_ids'],
                            'attention_mask': batch['attention_mask']}
            
            label_mask = batch['label_mask']

            optimizer.zero_grad()

            # complete sentence as input
            logits = model(**inputs_targets).logits
            # shifting in the logits
            shift_logits = logits[..., :-1, :].contiguous()
            # shifting in the labels
            shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
            
            # compute loss
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(bs, -1)
            loss = (loss * label_mask[:,1:]).sum(1) / label_mask[:,1:].sum(1)
            loss = loss.mean()

            accelerator.backward(loss)

            # This is the actual neurons freezing code
            if freeze:
                # For each layer
                for name, params in model.named_parameters():
                    # If in historical file freeze just stubborn neurons
                    if ".".join(name.split('.')[:-1]) in stubborn_idx:
                        # 1D layers
                        if len(params.shape) == 1:
                            params.grad[stubborn_idx[".".join(name.split('.')[:-1])]] = 0
                        # 2D layers (freeze columns of correspondent index)
                        else:
                            if freeze_other_layers:
                                params.grad[:,stubborn_idx[".".join(name.split('.')[:-1])]] = 0

                    # If not in historical file freeze all neurons (put gradients to zero)
                    else:
                        if len(params.shape) == 1:
                            params.grad[:] = 0
                        else:
                            params.grad[:,:] = 0
            
            # Perform update
            optimizer.step()

            cum_loss += loss.item()

            global_step += 1  # Increment global step
            progress_bar.update(1)

        # Evaluation at the end of each epoch
        if ev_epochs:
            epoch_acc_update.append(verify_acc(model, dataset_update, padding_mask, pad_token))
            epoch_acc_eval.append(verify_acc(model, dataset_eval, padding_mask, pad_token))

    # Model saving 
    if save_model:        
        model.save_pretrained(out_dir)

    torch.cuda.empty_cache()
    
    # Optionally, reset maximum memory statistics if needed
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    
    # Final evaluation
    score_update = verify_acc(model, dataset_update, padding_mask, pad_token)
    score_eval = verify_acc(model, dataset_eval, padding_mask, pad_token)

    if dataset_generality is not None:
        score_gen = verify_acc(model, dataset_generality, padding_mask, pad_token)
        if freeze:
            return epoch_acc_update, epoch_acc_eval, score_update, score_eval, score_gen, stubborn_number
        else:
            return epoch_acc_update, epoch_acc_eval, score_update, score_eval, score_gen
    else:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if freeze:
            return epoch_acc_update, epoch_acc_eval, score_update, score_eval, stubborn_number
        else:
            return epoch_acc_update, epoch_acc_eval, score_update, score_eval


'''
Stubborn neurons extraction
This function is used to extract stubborn neurons from historical data. Hisotrical data are taken and normalized over the whole model. Then neurons above the given threshold are
selected and inserted in the output.
'''
def stubborn_extraction(historical_path, norm, threshold, inverse=False, return_stubborn=True):

    # Load historical file
    with open(historical_path, 'rb') as file:
        data = pickle.load(file)

    # Choose the kind of data you want to use
    if norm:
        hist_data = data['grad_hist_dict']['norm']
    else:
        hist_data = data['grad_hist_dict']['raw']

    # Concatenate all gradients to have a complete ditribution
    grad_data = np.array([])

    for layer in hist_data:
        grad_data=np.concatenate((grad_data,hist_data[layer].flatten()))

    stubborn_grad = {}
    num_neurons = []
    count = 0
    
    # Free neurons
    if not inverse:
        # Sort all abs gradients in ascending order and select the element at position t as threshold
        t = np.sort(np.abs(grad_data))[threshold]

        for layer in hist_data:
            # Mask all neurons which are over the threshold
            if return_stubborn:  
                stubborn_grad[layer] = np.where(np.abs(hist_data[layer]) > t)
            # Do the opposite (return neurons to be updated)
            else:
                stubborn_grad[layer] = np.where(np.abs(hist_data[layer]) < t)

            num_neurons.append(hist_data[layer].shape[0])
            count += len(stubborn_grad[layer][0])

    # Busy neurons
    else:
        # Sort all abs gradients in desceding order and select the element at position t as threshold
        t = np.sort(np.abs(grad_data))[::-1][threshold]

        for layer in hist_data:
            # Mask all neurons which are lower than the threshold
            if return_stubborn:
                stubborn_grad[layer] = np.where(np.abs(hist_data[layer]) < t)
            # Do the opposite (return neurons to be updated)
            else:
                stubborn_grad[layer] = np.where(np.abs(hist_data[layer]) > t)
            num_neurons.append(hist_data[layer].shape[0])
            count += len(stubborn_grad[layer][0])

    return stubborn_grad, num_neurons, count


'''
Stubborn neurons extraction (plus)
This function is used to extract stubborn neurons from historical data using the intersection method to avoid touching specific neurons related to previous knowledge.
'''
def stubborn_extraction_plus(historical_path, historical_path_gen, threshold, return_stubborn=True):

    with open(historical_path, 'rb') as file:
        data_A = pickle.load(file)
    with open(historical_path_gen, 'rb') as file:
        data_H = pickle.load(file)

    hist_data_A = data_A['grad_hist_dict']['raw']
    hist_data_H = data_H['grad_hist_dict']['raw']

    grad_data_A = np.array([])

    for layer in hist_data_A:
        grad_data_A=np.concatenate((grad_data_A,hist_data_A[layer].flatten()))

    stubborn_grad = {}
    num_neurons = []
    count = 0

    # Sort all abs gradients in desceding order and select the element at position t as threshold
    t_A = np.sort(np.abs(grad_data_A))[::-1][threshold]
    masked_H ={}
    
    for layer in hist_data_A:
        # This is the set of busy_A
        update_A = np.where(np.abs(hist_data_A[layer]) > t_A)

        # Create a copy of hist H where all busy_A are set to -inf to avoid them for sure
        #masked_H[layer] = hist_data_H[layer]
        #masked_H[layer][update_A] = float('-inf')
        hist_data_H[layer][update_A] = 0
        #print(hist_data_H[layer].shape)

    grad_data_H = np.array([])

    for layer in hist_data_H:
        grad_data_H=np.concatenate((grad_data_H,hist_data_H[layer].flatten()))

    t_H = np.sort(np.abs(grad_data_H))[::-1][threshold]

    for layer in hist_data_A:

        #update_H = np.where(np.abs(hist_data_H[layer]) > t_H)

        # Mask all neurons which are lower than the threshold
        if return_stubborn:
            # stubborn_H = np.where(np.abs(hist_data_H[layer]) < t_H)
            stubborn_grad[layer] = np.where(np.abs(hist_data_H[layer]) < t_H)

            # spec_A = np.intersect1d(update_H[0], update_A[0])

            # stubborn_grad[layer] = (np.concatenate((stubborn_H[0],spec_A)),)

        # Do the opposite (return neurons to be updated)
        else:
            # update_H = np.where(np.abs(hist_data_H[layer]) > t_H)
            stubborn_grad[layer] = np.where(np.abs(hist_data_H[layer]) > t_H)
            # remove from the set of busy_H the element shared with busy_A
            # stubborn_grad[layer] = (np.setdiff1d(update_H[0],update_A[0]),)

        num_neurons.append(hist_data_H[layer].shape[0])
        count += len(stubborn_grad[layer][0])

    return stubborn_grad, num_neurons, count


'''
Random neurons generation
This function is used to generate a random distribution of stubborn neurons across the given layers with the same cardinality of the original threshold.
'''
def generate_rnd_distr(x, length, max_threshold):

    # Generate 'length' random numbers
    random_numbers = np.random.random(length)
    
    # Scale the numbers so that their sum is x
    scaled_numbers = random_numbers / np.sum(random_numbers) * x
    
    # Ensure no element exceeds the max_threshold
    result = np.minimum(scaled_numbers, max_threshold)
    
    # Convert to integers
    result = np.floor(result).astype(int)
    
    # Adjust to ensure the sum is exactly x
    current_sum = np.sum(result)
    difference = x - current_sum

    while difference > 0:
        indices = np.random.permutation(length)  # Shuffle indices
        for i in indices:
            if difference == 0:
                break
            if result[i] < max_threshold[i]:
                result[i] += 1
                difference -= 1
    
    return result.tolist()