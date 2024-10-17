import sys
# custom path insertion

sys.path.insert(1, '.')
from datasets import Dataset
import pandas as pd
import copy

'''
Counterfact dataset loading function
'''
def load_counterfact(path):
    dataset = pd.read_parquet(path)

    dataset = dataset.join(pd.json_normalize(dataset['requested_rewrite']))
    dataset = dataset.drop(columns=['requested_rewrite'])
    dataset['formatted_prompt']=dataset.apply(lambda row: row['prompt'].format(row['subject']), axis=1)
    dataset['formatted_fact_true'] = dataset['formatted_prompt'].str.cat(dataset['target_true.str'], sep=' ')
    dataset['formatted_fact_new'] = dataset['formatted_prompt'].str.cat(dataset['target_new.str'], sep=' ')
    
    dataset['rephrased_prompt'] = dataset['generation_prompts'].apply(lambda x: x[0])

    dataset['rephrased_fact_true'] = dataset['rephrased_prompt'].str.cat(dataset['target_true.str'], sep=' ')
    dataset['rephrased_fact_new'] = dataset['rephrased_prompt'].str.cat(dataset['target_new.str'], sep=' ')

    dataset=dataset[['case_id',
                'subject',
                'formatted_prompt',
                'formatted_fact_true',
                'formatted_fact_new',
                'rephrased_prompt',
                'rephrased_fact_true',
                'rephrased_fact_new',
                'target_true.str',
                'target_new.str']]

    dataset = Dataset.from_pandas(dataset)

    return dataset


'''
Tokenize function
This function implement the tokenization process for Counterfact dataset
'''
def tokenize_function(facts, tokenizer, counterfacts=False, rephrase=False, left=False):

    if left:
        pass
        # TO-DO: implement left padding
   
    else:
        # Tokenize counter facts
        if counterfacts:
            if rephrase:
                tok_facts = tokenizer(facts["rephrased_fact_new"], padding="longest", truncation=True)
            else:
                tok_facts = tokenizer(facts["formatted_fact_new"], padding="longest", truncation=True)
        # Toeknize true facts
        else:
            if rephrase:
                tok_facts = tokenizer(facts["rephrased_fact_true"], padding="longest", truncation=True)
            else:
                tok_facts = tokenizer(facts["formatted_fact_true"], padding="longest", truncation=True)
        
        # Tokenize prompt
        if rephrase:
            tok_prompts = tokenizer(facts["rephrased_prompt"], truncation=True, padding=False)
        else:
            tok_prompts = tokenizer(facts["formatted_prompt"], truncation=True, padding=False)
        
        # Modify labels starting from the input as target
        tok_facts["labels"] = copy.deepcopy(tok_facts['input_ids'])

        tok_facts["label_mask"] = []

        for i, l in enumerate(tok_facts['labels']):
            prompt_len = len(tok_prompts['input_ids'][i])
            # Insert -100 for prompts tokens (for trainer)
            l[:prompt_len] = [-100] * prompt_len
            # Genberate a boolean mask to consider just the answer (for custom ft)
            tok_facts['label_mask'].append([False] * prompt_len + [True] * (len(l)-prompt_len))


    return tok_facts