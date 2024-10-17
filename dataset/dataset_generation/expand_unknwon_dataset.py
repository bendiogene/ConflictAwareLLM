import json

"""
Expansion of unknown dataset.

This script is useful for preparing unknwon generated data to be processed by the tokenizer.

"""
with open('/home/MinnieMouse/project/epmem_edit/unknown_35.json') as file:
    data=json.load(file)


for i, elem in enumerate(data):
    data[i]["fact_id"]=i

print(len(data))

with open("/home/MinnieMouse/project/epmem_edit/dataset/multi_counterfact_unknown.json", "w") as json_file:
    json.dump(data, json_file, indent=4)