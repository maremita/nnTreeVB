from .seq_collections import SeqCollection

from abc import ABC, abstractmethod
import re

import numpy as np
import torch

__all__ = [
        'FullNucCatCollection',
        'build_categorical',
        'MSANucCatCollection',
        'build_msa_categorical']

__author__ = "amine"


nuc2cat = {
        'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 
        'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
        'U':[0.,0.,0.,1.], 'R':[.5,.5,0.,0.],
        'Y':[0.,0.,.5,.5], 'S':[0.,.5,.5,0.], 
        'W':[.5,0.,0.,.5], 'K':[0.,.5,0.,.5], 
        'M':[.5,0.,.5,0.], 'B':[0.,1/3,1/3,1/3], 
        'D':[1/3,1/3,0.,1/3], 'H':[1/3,0.,1/3,1/3],
        'V':[1/3,1/3,1/3,0.], 'N':[.25,.25,.25,.25],
        '-':[.25,.25,.25,.25]#, '?':[.25,.25,.25,.25]
        }

# pl = partial likelihood
nuc2pl = {
        'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 
        'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
        'U':[0.,0.,0.,1.], 'R':[1.,1.,0.,0.],
        'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 
        'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.], 
        'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 
        'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
        'V':[1.,1.,1.,0.], 'N':[1.,1.,1.,1.],
        '-':[1.,1.,1.,1.]#, '?':[1.,1.,1.,1.]
        }

# #####
# Base collections
# ################

class BaseCollection(ABC):

    def __compute_rep_from_collection(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_rep_of_sequence(seq.seq._data, i)
            self.ids.append(seq.id)

        return self

    def __compute_rep_from_strings(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_rep_of_sequence(seq, i)
            self.ids.append(i)

        return self
 
    def _compute_representations(self, sequences):
        if isinstance(sequences, SeqCollection):
            self.__compute_rep_from_collection(sequences)

        else:
            self.__compute_rep_from_strings(sequences)

    @abstractmethod
    def _compute_rep_of_sequence(self, seq, ind):
        """
        """

    # TODO Check for list of SeqCollection
    def check_sequences(self, seqs):
        sequences = []
        if isinstance(seqs, str):
            sequences = [seqs]
        elif isinstance(seqs, list):
            for seq in seqs:
                if isinstance(seq, str):
                    sequences.append(seq)
                else: print("Input object {} is not a string".format(seq))
        elif not isinstance(seqs, SeqCollection):
            raise("Input sequences should be string, list of string or SeqCollection")

        else : sequences = seqs

        return sequences


class FullNucCatCollection(BaseCollection):

    def __init__(self, sequences, nuc_cat=True, dtype=np.float32):

        if nuc_cat:
            self.nuc2rep = nuc2cat
        else:
            self.nuc2rep = nuc2pl

        self.dtype = dtype
        self.alphabet = "".join(self.nuc2rep.keys())
        sequences = self.check_sequences(sequences)
        #
        self.ids = []
        self.data = []
        self._compute_representations(sequences)

    def _compute_rep_of_sequence(self, sequence, ind):
        # ind is not used here
        sequence = sequence.upper()
        seq_array = list(re.sub(r'[^'+self.alphabet+']', 'N',
            sequence, flags=re.IGNORECASE))
        seq_cat = np.array([self.nuc2rep[i] for i in seq_array],
                dtype=self.dtype)
        self.data.append(seq_cat)

        return self


class MSANucCatCollection(BaseCollection):

    def __init__(self, sequences, nuc_cat=True, dtype=np.float32):
 
        if nuc_cat: 
            self.nuc2rep = nuc2cat
        else:
            self.nuc2rep = nuc2pl

        self.dtype = dtype
        self.alphabet = "".join(self.nuc2rep.keys())
        sequences = self.check_sequences(sequences)
        #
        self.msa_len = len(sequences[0])
        self.nbseqs = len(sequences)
        self.ids = []
        self.data = np.zeros((self.msa_len, self.nbseqs,4 ), dtype=self.dtype)
        self._compute_representations(sequences)

    def _compute_rep_of_sequence(self, sequence, ind):
        sequence = sequence.upper()
        seq_array = list(re.sub(r'[^'+self.alphabet+']', 'N',
            sequence, flags=re.IGNORECASE))

        assert len(seq_array) == self.msa_len

        for i, ind_char in enumerate(seq_array):
            self.data[i, ind] = self.nuc2rep[ind_char]

        return self

# #####
# Data build functions
# ####################

def build_categorical(
        seq_data,
        nuc_cat=True,
        dtype=np.float32):

    return FullNucCatCollection(
            seq_data, 
            nuc_cat=nuc_cat, 
            dtype=dtype)

def build_msa_categorical(
        seq_data,
        nuc_cat=True,
        dtype=np.float32):

    return MSANucCatCollection(
            seq_data,
            nuc_cat=nuc_cat,
            dtype=dtype)
