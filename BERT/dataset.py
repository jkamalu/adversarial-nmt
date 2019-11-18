''' General dataset class
'''

import torch
from ..transformers.transformers import BertTokenizer
import os
import logging

from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_folder, is_train=True):
        '''
        Requires passing in a folder to where the dataset is stored
        '''
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # data is stored as a list of tuples of strings (l1-l2)
        self.parallel_data = self.load_data(data_folder, is_train)

    @staticmethod
    def load_data(data_folder, is_train):
        for filename in os.listdir(os.path.join("data", data_folder)):
            logging.info("Found file with name {filename}".format(filename=filename))


    def __len__(self):
        return len(self.parallel_data)

    def __getitem__(self, idx):
        return self.parallel_data[idx]
