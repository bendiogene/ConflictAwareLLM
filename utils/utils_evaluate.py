import sys
# custom path insertion
sys.path.insert(1, '.')
from collections.abc import Sequence
import gc

import torch

'''
Verify accuracy
External loop to verify accuracy on each batch of the evaluation dataset.
'''
def verify_acc(model, dataset, padding_mask, pad_token, get_idx=False, batch_size=500):

    if len(dataset) == 0:
        return 0

    device = model.device
    
    model.eval()

    correct_facts = 0

    iter=dataset.iter(batch_size=batch_size)
    known_idx = []

    # For each batch run accuracy tests
    for i,batch in enumerate(iter):

        start_sample = i*batch_size
        # Extract batch metrics
        cf, ki= test_prediction_acc(model, batch, device, padding_mask, pad_token, start_sample)
        # Increment correct facts counter
        correct_facts+=cf
        # Extract known facts idxs
        known_idx.extend(ki)

    if get_idx:
        return correct_facts/len(dataset), known_idx
    else:  
        return correct_facts/len(dataset)


'''
Accuarcy computation
Internal accuracy computation. The concept is to verify in a single pass that the 
'''
def test_prediction_acc(model, batch, device, padding_mask=False, pad_token=50726, start_sample=0):

    with torch.no_grad():
        inputs_targets={'input_ids': torch.tensor(batch['input_ids']).to(device),'attention_mask': torch.tensor(batch['attention_mask']).to(device)}
        
       
        outputs = model(**inputs_targets)

        del inputs_targets
        gc.collect()
        torch.cuda.empty_cache()

        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        
        # Generate the answer
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()

        labels = batch['labels']

        # Extract prompt length (left) and padding length (right)
        prompt_len = [l.count(-100) for l in labels]
        padding_len = [l.count(pad_token) for l in labels]

        # Perform the list sliding to have matching positions for answers and labels. Prompt is not considered in the matching computation.
        # Consider padding
        if not padding_mask:
            answers = slice_list(answers, prompt_len, padding_len, padding_mask=False, left=True)
            labels = slice_list(labels, prompt_len, padding_len, padding_mask=False, left=False)
        # Mask padding
        else:
            answers = slice_list(answers, prompt_len, padding_len, padding_mask=True, left=True)
            labels = slice_list(labels, prompt_len, padding_len, padding_mask=True, left=False)

        counter = 0
        known_idx = []
        for i, (row1, row2) in enumerate(zip(answers, labels)): 

            if row1 == row2:
                known_idx.append(start_sample + i)
                counter += 1

        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return counter, known_idx
        

'''
List sliding
Function to slide and cut lists to have a proper matching between target and predictions.
'''
def slice_list(matrix, start_indices, pad_sizes, padding_mask, left):
    if not isinstance(matrix[0], Sequence):
        matrix = [matrix]

    if not isinstance(start_indices, Sequence):
        start_indices = [start_indices]

    if not isinstance(pad_sizes, Sequence):
        pad_sizes = [pad_sizes]   

    if not padding_mask:
            if left:
                return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
            else:
                return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
            
    else:
            if left:
                return [row[start_index-1:-pad_size-1] for row, start_index, pad_size in zip(matrix, start_indices, pad_sizes)]
            else:
                return [row[start_index:-pad_size] for row, start_index, pad_size in zip(matrix, start_indices, pad_sizes)]
