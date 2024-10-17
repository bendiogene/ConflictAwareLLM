from datasets import Dataset
import pandas as pd
import json

dataset = pd.read_parquet("dataset/facts/counterfact_dataset/train.parquet")

dataset = dataset.join(pd.json_normalize(dataset['requested_rewrite']))
dataset = dataset.drop(columns=['requested_rewrite'])
dataset['formatted_prompt']=dataset.apply(lambda row: row['prompt'].format(row['subject']), axis=1)
dataset['formatted_fact_true'] = dataset['formatted_prompt'].str.cat(dataset['target_true.str'], sep=' ')
dataset['formatted_fact_new'] = dataset['formatted_prompt'].str.cat(dataset['target_new.str'], sep=' ')

dataset=dataset[['case_id',
                'formatted_prompt',
                'formatted_fact_true',
                'formatted_fact_new',
                'target_true.id',
                'target_new.id',
                'generation_prompts'
                ]]

with open('/home/MinnieMouse/project/epmem_edit/dataset/facts/counterfact.json') as file:
    dataset_json = json.load(file)

print(dataset['case_id'][999])
print(dataset_json[999]['case_id'])

dataset_json_filtered = []

n = 0

for i,row in dataset.iterrows():
   

    k = i + n
    while row['case_id'] != dataset_json[k]['case_id']:
        n += 1
        k = i + n

    print(f"{row['case_id']} - {dataset_json[k]['case_id']}")
    dataset_json_filtered.append(dataset_json[k])

print(len(dataset_json_filtered))

with open('/home/MinnieMouse/project/epmem_edit/dataset/facts/counterfact_filtered.json', 'w') as file:
     json.dump(dataset_json_filtered, file, indent=4)

