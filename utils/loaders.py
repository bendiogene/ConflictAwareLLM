from abc import ABC
import pandas as pd
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np


class FamiliarityClassificationDataset(data.Dataset, ABC):
    '''
        FamiliarityClassification Dataset

        Dataset built in order to classify the familiarity of facts. It is built in order to load gradients and activations features from a .pkl file.
    '''
    def __init__(self, split, modalities, mode, layers, dataset_path, processing_1='masked_avg',processing_2='flatten', **kwargs):

        self.modalities = modalities  # considered modalities ([gradients, activations])
        self.mode = mode  # 'train', 'val' or 'test'
        self.layers = layers # number of layers
        self.dataset_path = dataset_path # path of the dataset file
        self.split = split # dataset split (ex. first_1000)
        self.processing_1 = processing_1
        self.processing_2 = processing_2

        # creation of the pickle file name considering the split and the modality (e.g. D1 + _ + test + .pkl)
        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        else:
            pickle_name = split + "_test.pkl"

        # get the pickle file location (path + name). The path must be inserted in dataset_conf.annotations_path
        self.actions_list = pd.read_pickle(os.path.join(self.dataset_path, pickle_name))
        print(f"Dataloader for {split}-{self.mode} with {len(self.actions_list)} samples generated")

        self.features = pd.DataFrame(pd.read_pickle(os.path.join(dataset_path,pickle_name)))[
                ["fact_id", "activations", "gradients", "class_id", "answer_tokens_idx"]]

    def __getitem__(self, index):
        # record is a row of the pkl file containing one sample/action
        output = {}
        sample_row = self.features.iloc[index]


        for m in self.modalities:
            features = sample_row[m]
            out=None

            if self.processing_1 == 'masked_avg':
                features_sliced = features[:,sample_row["answer_tokens_idx"],:]
                features_avg = np.mean(np.abs(features_sliced), axis=1)
                out = features_avg
            else:
                out = sample_row[m]

            if self.processing_2 == 'flatten':
                out=out.flatten()
            elif self.processing_2 == 'key':
                out=out[:,768:].flatten()

            output[m] = torch.tensor(out)

        output['label'] = sample_row['class_id']

        return output

    def __len__(self):
        return len(self.actions_list)
    


class AVGActivationsDataset(data.Dataset, ABC):
    '''
        FamiliarityClassification Dataset

        Dataset built in order to classify the familiarity of facts. It is built in order to load gradients and activations features from a .pkl file.
    '''
    def __init__(self, split, mode, layers, dataset_path, **kwargs):

        self.mode = mode  # 'train', 'val' or 'test'
        self.layers = layers # number of layers
        self.dataset_path = dataset_path # path of the dataset file
        self.split = split # dataset split (ex. first_1000)

        # creation of the pickle file name considering the split and the modality (e.g. D1 + _ + test + .pkl)
        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        else:
            pickle_name = split + "_test.pkl"

        # get the pickle file location (path + name). The path must be inserted in dataset_conf.annotations_path
        self.actions_list = pd.read_pickle(os.path.join(self.dataset_path, pickle_name))
        print(f"Dataloader for {split}-{self.mode} with {len(self.actions_list)} samples generated")

        self.features = pd.DataFrame(pd.read_pickle(os.path.join(dataset_path,pickle_name)))[
                ["mlp_activations_norm","attn_activations_norm", "mlp_gradients_norm","attn_gradients_norm", "class_id"]]

    def __getitem__(self, index):
        # record is a row of the pkl file containing one sample/action
        output = {}
        sample_row = self.features.iloc[index]

        output['activations'] = np.concatenate((sample_row['mlp_activations_norm'],sample_row['attn_activations_norm']))
        output['gradients'] = np.concatenate((sample_row['mlp_gradients_norm'],sample_row['attn_gradients_norm']))
        output['label'] = sample_row['class_id']

        return output

    def __len__(self):
        return len(self.actions_list)
    



class AccuracyDataset(data.Dataset, ABC):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['question']
        answer = item['answer']
        return question, answer
    

class FactDataset(data.Dataset, ABC):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_target = item['input_target']
        label_mask = item['label_mask']
        return {'input_target_ids':input_target['input_ids'],
                'input_target_mask':input_target['attention_mask'],
                'label_mask': label_mask}