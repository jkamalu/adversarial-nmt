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
    def __init__(self, data_folder, is_train=True):
        '''
        Requires passing in a folder to where the dataset is stored
        '''

        # data is stored as a list of tuples of strings (l1-l2)
        self.parallel_data = self.load_data(data_folder, data_type="train" if is_train else "val")
        self.onmt_vocab = self.load_onmt_vocab(data_folder)

    @staticmethod
    def load_data(data_folder, data_type="train"):
        parallel_data = []
        root_directory_path = os.path.join(os.getcwd(), data_folder)

        removed_counter = 0

        if (data_type != "train" and data_type != "val"):
             raise ValueError("data_type needs to be either set to train or val")

        for filename in os.listdir(root_directory_path):

            if (data_type == "train"):
                if('.train' not in filename):
                     continue
            else:
                if('.val' not in filename):
                     continue

            print("Loading data from file: {filename}".format(filename=filename))

            _torch_dataset = torch.load(os.path.join(root_directory_path,filename))

            for idx, _example in enumerate(_torch_dataset):

                _src = _example.src
                _tgt = _example.tgt
                _idx = _example.indices

                if (len(_src[0]) < 2 or len(_tgt[0]) < 2):
                    removed_counter += 1
                    continue

                parallel_data.append((_src[0], _idx, _tgt[0]))

        print("removed {} examples - not long enough".format(removed_counter))
        return parallel_data


    @staticmethod
    def load_onmt_vocab(data_folder):
        root_directory_path = os.path.join(os.getcwd(), data_folder)
        for filename in os.listdir(root_directory_path):
            if ('.vocab' not in filename):
                 continue
            _vocab = torch.load(os.path.join(root_directory_path,filename))
            src_vocab = _vocab['src'].fields[0][1].vocab.itos
            tgt_vocab = _vocab['tgt'].fields[0][1].vocab.itos
            return (src_vocab, tgt_vocab)
        raise Exception("Was unable to find a vocab file in the data directory.")


    def __len__(self):
        return len(self.parallel_data)

    def __getitem__(self, idx):
        '''Returning token indices '''
        return (self.parallel_data[idx][0], self.parallel_data[idx][2])
