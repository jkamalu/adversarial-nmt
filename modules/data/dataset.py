__author__ = 'Richard Diehl Martinez, John Kamalu'

import os
import pickle
import logging

from tqdm import tqdm_notebook as tqdm

import torch
from torch.utils.data import Dataset


class Collator(object):

    def __init__(self, maxlen):
        self.maxlen = maxlen

    def __call__(self, data):
        sentences_l1, sentences_l2 = zip(*data)

        lengths_l1 = [len(sentence) for sentence in sentences_l1]
        lengths_l2 = [len(sentence) for sentence in sentences_l2]

        batch_size = len(sentences_l1)

        idx_tensor_l1 = torch.zeros((batch_size, self.maxlen), dtype=torch.long)
        idx_tensor_l2 = torch.zeros((batch_size, self.maxlen), dtype=torch.long)

        idx_tensor_no_eos_l1 = torch.zeros((batch_size, self.maxlen - 1), dtype=torch.long)
        idx_tensor_no_eos_l2 = torch.zeros((batch_size, self.maxlen - 1), dtype=torch.long)

        for idx, (sentence_len, sentence_l1) in enumerate(zip(lengths_l1, sentences_l1)):
            idx_tensor_l1[idx, :] = torch.tensor(sentence_l1 + [1]*(self.maxlen - sentence_len))
            idx_tensor_no_eos_l1[idx, :] = torch.tensor(sentence_l1[:-1] + [1] * (self.maxlen - sentence_len))

        for idx, (sentence_len, sentence_l2) in enumerate(zip(lengths_l2, sentences_l2)):
            idx_tensor_l2[idx, :] = torch.tensor(sentence_l2 + [1]*(self.maxlen - sentence_len))
            idx_tensor_no_eos_l2[idx, :] = torch.tensor(sentence_l2[:-1] + [1] * (self.maxlen - sentence_len))
            
        return (
            (idx_tensor_l1, idx_tensor_no_eos_l1, torch.tensor(lengths_l1)),
            (idx_tensor_l2, idx_tensor_no_eos_l2, torch.tensor(lengths_l2))
        )


class TextDataset(Dataset):
    
    def __init__(self, directory, tokenizer_en, tokenizer_fr, training=True, minlen=2, maxlen=50, size=-1):

        # data is stored as a list of tuples of strings (l1-l2)
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr

        mode = "train" if training else "val"
        self.parallel_data = self.load_data(directory, mode, minlen, maxlen, size)

    def load_data(self, directory, mode, minlen, maxlen, size):

        parallel_data = []
        removed_count_short = 0
        removed_count_long = 0
        read_counter = 0

        if mode not in  ["train", "val"]:
             raise ValueError("mode needs to be either \'train\' or \'val\'.")

        directory = os.path.join(os.getcwd(), directory)
        pkl_file = list(filter(lambda fd: fd.endswith(mode + ".pkl"), os.listdir(directory)))
        if pkl_file:
            # Loading in from stored out pkl files
            load_in_file = os.path.join(directory, pkl_file[0])
            return pickle.load(open(load_in_file, "rb"))

        files = filter(lambda fd: fd.endswith(mode), os.listdir(directory))
        files = {f.split('.')[-2]:os.path.join(directory, f) for f in files}

        with open(files["en"], "rt") as en, open(files["fr"], "rt") as fr:
            while True:
                
                if (read_counter % 10000 == 0):
                    print("{} examples processed.".format(read_counter))
                    
                read_counter += 1
                line_en = en.readline()
                line_fr = fr.readline()

                if line_en == "" or line_fr == "":
                    break

                line_en = line_en.strip().lower()
                line_fr = line_fr.strip().lower()

                # splitting and doing this check first speeds up computation
                too_short = len(line_en.split()) < minlen or len(line_fr.split()) < minlen
                too_long = len(line_en.split()) > maxlen or len(line_fr.split()) > maxlen

                if too_long:
                    removed_count_long += 1
                    continue

                if too_short:
                    removed_count_short += 1
                    continue

                tokenized_en = self.tokenizer_en.encode(line_en)
                tokenized_fr = self.tokenizer_fr.encode(line_fr)

                too_short = len(tokenized_en) < minlen or len(tokenized_fr) < minlen
                too_long = len(tokenized_en) > maxlen or len(tokenized_fr) > maxlen

                if too_long:
                    removed_count_long += 1
                    continue

                if too_short:
                    removed_count_short += 1
                    continue

                parallel_data.append((tokenized_en, tokenized_fr))
                
                if len(parallel_data) >= size and size > 0:
                    break

        print("# examples with length < {} removed: {}".format(minlen, removed_count_short))
        print("# examples with length > {} removed: {}".format(maxlen, removed_count_long))

        saved_out_file = open(os.path.join(directory, "data.{}.pkl".format(mode)), 'wb')
        pickle.dump(parallel_data, saved_out_file)

        return parallel_data

    def __len__(self):
        return len(self.parallel_data)

    def __getitem__(self, idx):
        return self.parallel_data[idx]
