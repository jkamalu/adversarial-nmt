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

class Collator(object):
    ''' colltor object which can be called by the dataloader class for processing batches.'''
    def __init__(self, maxlen=50,):
        self.max_len = maxlen # standardized length

    def __call__(self, data):
        input_sentences, output_sentences = zip(*data)

        input_lengths = [len(sentence) for sentence in input_sentences]
        output_lengths = [len(sentence) for sentence in output_sentences]

        batch_size = len(input_sentences)

        input_idx_tensor = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        output_idx_tensor = torch.zeros((batch_size, self.max_len), dtype=torch.long)

        input_idx_tensor_no_eos = torch.zeros((batch_size, self.max_len-1), dtype=torch.long)
        output_idx_tensor_no_eos = torch.zeros((batch_size, self.max_len-1), dtype=torch.long)


        for idx, (sentence_len, input_sentence) in enumerate(zip(input_lengths, input_sentences)):
            input_idx_tensor[idx, :] = torch.tensor(input_sentence + [1]*(self.max_len-sentence_len))
            input_idx_tensor_no_eos[idx, :] = torch.tensor(input_sentence[:-1] + [1]*(self.max_len-sentence_len))


        for idx, (sentence_len, output_sentence) in enumerate(zip(output_lengths, output_sentences)):
            output_idx_tensor[idx, :] = torch.tensor(output_sentence + [1]*(self.max_len-sentence_len))
            output_idx_tensor_no_eos[idx, :] = torch.tensor(output_sentence[:-1] + [1]*(self.max_len-sentence_len))

        return ((input_idx_tensor, input_idx_tensor_no_eos, torch.tensor(input_lengths)), (output_idx_tensor, output_idx_tensor_no_eos, torch.tensor(output_lengths)))


class TextDataset(Dataset):
    def __init__(self, directory,
                       tokenizer_en,
                       tokenizer_fr,
                       training=True,
                       minlen=2,
                       maxlen=50):
        '''
        Requires passing in a folder to where the dataset is stored
        '''

        # data is stored as a list of tuples of strings (l1-l2)
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr

        mode = "train" if training else "val"
        self.parallel_data = self.load_data(directory, mode, minlen, maxlen)

    def load_data(self, directory, mode, minlen, maxlen):

        parallel_data = []
        removed_count_short = 0
        removed_count_long = 0
        read_counter = 0

        if mode not in  ["train", "val"]:
             raise ValueError("mode needs to be either \'train\' or \'val\'.")

        directory = os.path.join(os.getcwd(), directory)
        files = filter(lambda fd: fd.endswith(mode), os.listdir(directory))
        files = {f.split('.')[-2]:os.path.join(directory, f) for f in files}

        with open(files["en"], "rt") as en, open(files["fr"], "rt") as fr:
            while True:
                if (read_counter > limit_data):
                    break

                if (read_counter % 10000 == 0):
                    print(read_counter)
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

        print("{} examples with length < {} removed.".format(removed_count_short, minlen))
        print("{} examples with length > {} removed.".format(removed_count_long, maxlen))

        print("Saving out ")
        import pickle
        filehandler = open("train_dataset.pkl", 'wb')
        pickle.dump(self, filehandler)

        return parallel_data

    def __len__(self):
        return len(self.parallel_data)

    def __getitem__(self, idx):
        '''Returning token indices '''
        return self.parallel_data[idx]
