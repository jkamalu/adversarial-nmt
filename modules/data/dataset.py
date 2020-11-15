__author__ = 'John Kamalu'

import os
import csv
import tempfile
import logging
from functools import partial

import numpy as np

import pandas

import h5py

import torch
from torch.utils.data import IterableDataset

class EuroparlDataset(IterableDataset):

    def __init__(self, fname, tokenizer_l1=None, tokenizer_l2=None):
        super().__init__()

        self.fname = fname
        self.tokenizer_l1 = tokenizer_l1
        self.tokenizer_l2 = tokenizer_l2
        
        with h5py.File(fname, 'r') as hdf5:
            self.length = len(hdf5["l1"]["seq"])
        
        self.initialized = False
        
    def _init_singleton(self):
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if not self.initialized:
            self.seq_l1 = h5py.File(self.fname, "r")["l1"]["seq"]
            self.seq_l2 = h5py.File(self.fname, "r")["l2"]["seq"]
            self.len_l1 = h5py.File(self.fname, "r")["l1"]["len"]
            self.len_l2 = h5py.File(self.fname, "r")["l2"]["len"]

            assert len(self.seq_l1) == len(self.len_l1) and \
                   len(self.seq_l1) == len(self.len_l2) and \
                   len(self.seq_l1) == len(self.seq_l2)
            
            self.initialized = True

    def __iter__(self):
        self._init_singleton()
        
        return map(self._to_tensor,
            zip(
                zip(
                    self.seq_l1.__iter__(), 
                    map(lambda x: self._del_eos(*x), zip(self.seq_l1.__iter__(), self.len_l1.__iter__())), 
                    self.len_l1.__iter__()
                ),
                zip(
                    self.seq_l2.__iter__(), 
                    map(lambda x: self._del_eos(*x), zip(self.seq_l2.__iter__(), self.len_l2.__iter__())), 
                    self.len_l2.__iter__()
                )
            )
        )

    def __len__(self):
        return self.length

    def __getitem__(self, *args):
        self._init_singleton()
        
        seq_l1 = self.seq_l1.__getitem__(*args)
        seq_l2 = self.seq_l2.__getitem__(*args)
        
        len_l1 = self.len_l1.__getitem__(*args)
        len_l2 = self.len_l2.__getitem__(*args)
        
        return self._to_tensor([
            [seq_l1, self._del_eos(seq_l1, len_l1), len_l1],
            [seq_l2, self._del_eos(seq_l2, len_l2), len_l2]
        ])

    @classmethod
    def to_hdf5(cls, dirname, fname_l1, fname_l2, tokenizer_l1, tokenizer_l2, min_length, max_length, overwrite=False):
        
        basename = "{}-{}.min{}_max{}.hdf5".format(
            fname_l1.split(".")[-1],
            fname_l2.split(".")[-1],
            str(min_length).zfill(3),
            str(max_length).zfill(3)
        )
        fname = os.path.join(dirname, basename)

        if os.path.isfile(fname) and not overwrite:
            return cls(fname, tokenizer_l1=tokenizer_l1, tokenizer_l2=tokenizer_l2)
        elif os.path.isfile(fname):
            os.system(f"rm {fname}")
                
        with tempfile.TemporaryFile("w+t") as tmp_l1, tempfile.TemporaryFile("w+t") as tmp_l2:
            csv_writer_l1 = csv.writer(tmp_l1, delimiter=' ')
            csv_writer_l2 = csv.writer(tmp_l2, delimiter=' ')

            len_l1 = []
            len_l2 = []
            for seq_len_l1, seq_len_l2 in cls._stream(fname_l1, fname_l2, 
                                                      tokenizer_l1, tokenizer_l2, min_length, max_length):
                csv_writer_l1.writerow(seq_len_l1[0])
                csv_writer_l2.writerow(seq_len_l2[0])
                len_l1.append(seq_len_l1[1])
                len_l2.append(seq_len_l2[1])
                if len(len_l1) % 100000 == 0:
                    print(f"# encoded pairs: {len(len_l1)}")
            
            tmp_l1.seek(0)
            tmp_l2.seek(0)

            seq_l1 = pandas.read_csv(tmp_l1, sep=" ", header=None).values
            seq_l2 = pandas.read_csv(tmp_l2, sep=" ", header=None).values

        hdf5 = h5py.File(fname, mode="w")
        hdf5.create_dataset("l1/len", data=len_l1)
        hdf5.create_dataset("l2/len", data=len_l2)
        hdf5.create_dataset("l1/seq", data=seq_l1)
        hdf5.create_dataset("l2/seq", data=seq_l2)
        hdf5.close()
        
        return cls(fname, tokenizer_l1=tokenizer_l1, tokenizer_l2=tokenizer_l2)

    @classmethod
    def _stream(cls, fname_l1, fname_l2, tokenizer_l1, tokenizer_l2, min_length, max_length):
        reader_l1 = open(fname_l1)
        reader_l2 = open(fname_l2)
        mapper_l1 = map(partial(cls._tokenize, tokenizer_l1, max_length), reader_l1)
        mapper_l2 = map(partial(cls._tokenize, tokenizer_l2, max_length), reader_l2)
        zipper = zip(mapper_l1, mapper_l2)
        length = lambda x: len(x[0][0]) >= min_length and len(x[1][0]) >= min_length and \
                           len(x[0][0]) == max_length and len(x[1][0]) == max_length
        stream = filter(length, zipper)
        return stream

    @staticmethod
    def _tokenize(tokenizer, max_length, line):
        encoding = tokenizer.encode(line.lower().strip())
        return encoding + [tokenizer.pad_token_id] * (max_length - len(encoding)), len(encoding)
    
    @staticmethod
    def _del_eos(sequence, length):
        return np.delete(sequence, length - 1)

    @staticmethod
    def _to_tensor(item):
        return list(map(lambda i: list(map(torch.tensor, i)), item))