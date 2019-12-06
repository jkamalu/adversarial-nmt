__author__ = 'Richard Diehl Martinez'
'''
General dataset class. Reads in the torch TextDataset and stores the data
in the form of a more basic torch Dataset class. Eventually, we want to
change this back to a torch TextDataset class.
'''

import torch
import os
import logging
from tqdm import tqdm_notebook as tqdm

from torch.utils.data import Dataset

def collate(data):
    ''' Helper function passed into Dataloader class to specify collation'''
    return None

class TextDataset(Dataset):
    def __init__(self, directory, training=True, minlen=2):
        '''
        Requires passing in a folder to where the dataset is stored
        '''

        # data is stored as a list of tuples of strings (l1-l2)
        mode = "train" if training else "val"
        self.parallel_data = self.load_data(directory, mode, minlen)

    @staticmethod
    def load_data(directory, mode, minlen):

        parallel_data = []
        removed_count = 0

        if mode not in  ["train", "val"]:
             raise ValueError("mode needs to be either \'train\' or \'val\'.")
           
        directory = os.path.join(os.getcwd(), directory)
        files = filter(lambda fd: fd.endswith(mode), os.listdir(directory))
        files = {f.split('.')[-2]:os.path.join(directory, f) for f in files}
        
        with open(files["en"], "rt") as en, open(files["fr"], "rt") as fr:
            while True:
                line_en = en.readline()
                line_fr = fr.readline()
                
                if line_en == "" or line_fr == "":
                    break
                
                line_en = line_en.strip().lower()
                line_fr = line_fr.strip().lower()
                
                if len(line_en.split(" ")) < minlen or len(line_fr.split(" ")) < minlen:
                    removed_count += 1
                    continue
                
                parallel_data.append((line_en, line_fr))

        print(f"{removed_count} examples with length < {minlen} removed.")

        return parallel_data

    def __len__(self):
        return len(self.parallel_data)

    def __getitem__(self, idx):
        '''Returning token indices '''
        return self.parallel_data[idx]
