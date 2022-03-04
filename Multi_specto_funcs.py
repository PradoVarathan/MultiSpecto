


import argparse
import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import h5py

def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.
    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output
    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize
    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_)

    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

    n = 0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = mydict[c]
        n = n + 1

    # get the complementary sequences
    dataflip = seqsnp[:, ::-1, ::-1]
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp


def get_predicted_diff(snp_comb_seq,inputsize = 2000, batchSize = 32, maxshift = 800, args_cuda = False):
    """
    Function to obtain all the predicted chromatin values for reference and alterante 
    and find the difference among them for further analysis.
    Args:
        snp_comb_seq: A dictionary of sequences as string object with A,T,G,C characters
                        and keys corresponding to snps and combinations of snps with atleast
                        one snp having 'Ref' in the key name to denote reference variant
    Return:
            A dictionary of matrix size 4000x2002 for the chromatin difference values for each 
            variant and combination except the reference
    """
    refseqs = [seq for key, seq in snp_comb_seq.items() if 'ref' in key.lower()]
    ref_encoded = encodeSeqs(refseqs, inputsize=inputsize).astype(np.float32)

    ref_preds = []
    for i in range(int(1 + (ref_encoded.shape[0]-1) / batchSize)):
        input = torch.from_numpy(ref_encoded[int(i*batchSize):int((i+1)*batchSize),:,:]).unsqueeze(2)
        if args_cuda:
            input = input.cuda()
        ref_preds.append(model.forward(input).cpu().detach().numpy().copy())
    ref_preds = np.vstack(ref_preds)
    
    comb_diff_pred = {}
    for comb_seq in snp_comb_seq.keys():

        if('Ref' not in comb_seq):

            altseqs = [snp_comb_seq[comb_seq]]
            alt_encoded = encodeSeqs(altseqs, inputsize=inputsize).astype(np.float32)

            alt_preds = []
            for i in range(int(1 + (alt_encoded.shape[0]-1) / batchSize)):
                input = torch.from_numpy(alt_encoded[int(i*batchSize):int((i+1)*batchSize),:,:]).unsqueeze(2)
                if args_cuda:
                    input = input.cuda()
                alt_preds.append(model.forward(input).cpu().detach().numpy().copy())
            alt_preds = np.vstack(alt_preds)

            diff = alt_preds - ref_preds
            comb_diff_pred[comb_seq] = diff
    
    
    return comb_diff_pred