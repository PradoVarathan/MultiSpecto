## Required Modules

import pandas as pd
import numpy as np
import Bio
import os
from Bio import Entrez, SeqIO
import itertools
import argparse
import math
import xmltodict
from pprint import pprint
import torch
from torch import nn
import h5py
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import time
from itertools import compress
import pickle

## Required Functions and Classes
#Defining a SNP class to perform simple LD filtering duties
class SNP:
    
    def __init__(self,rsid,position,chromosome):
        self.rsid = rsid
        self.position = position
        self.chr = chromosome
    

        
    def check_ld_snps(self,dataset,window = 1000):
        start_position = self.position - window + 1
        end_position = self.position + window
        dataset = dataset[dataset['Chromosome'] == self.chr]
        def extract_neighbour_snps(start_position, end_position, dataset):
            neighbour_snps = []
            for index,row in dataset.iterrows():
                if start_position <= dataset.loc[index,'Position'] <= end_position:
                    neighbour_snps.append(dataset.loc[index,'MarkerName'])
                else:
                    continue
            return neighbour_snps
    
        self.snps_in_window = extract_neighbour_snps(start_position,end_position,dataset)
        return self.snps_in_window
    
    def obtain_snp_sequence(self,dataset,window = 1000):
        start_position = self.position - window +1
        end_position = self.position + window
        if int(self.chr) < 10:
            id_chr = "".join(["NC_00000",str(self.chr)])
        else:
            id_chr = "".join(["NC_0000",str(self.chr)])

        handle = Entrez.efetch(db="nucleotide",
                        id = id_chr,
                        rettype = "fasta",
                        strand = 1,
                        seq_start = start_position,
                        seq_stop  = end_position)
        record = SeqIO.read(handle,"fasta")
        Seq_temp = str(record.seq)
        idx = dataset['MarkerName'] == self.rsid
        allele = str(dataset.loc[idx,'Effect_allele'].values[0])
        self.snp_sequence = Seq_temp[:999] + allele + Seq_temp[1000:]
        return self.snp_sequence
    
    def obtain_all_comb_seq(self,dataset,sign_num = 'null', window = 1000):
        
        def all_snp_combinations(a):
            combinations = []
            for k in range(0,len(a)):
                t = list(itertools.combinations(a,k+1))
                combinations.extend(t)
            return combinations
        
        self.combinations = all_snp_combinations(self.snps_in_window)
        comb_names = ['_'.join(x) for x in self.combinations if len(x)> 0]
        comb_names.append('_'.join(['Ref',self.rsid]))
        combination_dataset = dataset[dataset['MarkerName'].isin(self.snps_in_window)]
        if sign_num != 'null':
            combination_dataset = combination_dataset.sort_values('Pvalue')
            combination_dataset = combination_dataset.iloc[0:int(sign_num),:]
        sequences = []
        
        for comb in self.combinations:
            seq_to_change = self.snp_sequence
            start_position = self.position - window + 1
            end_position = self.position + window
            for k in range(0,len(comb)):
                idx = combination_dataset['MarkerName'] == comb[k]
                pos = combination_dataset.loc[idx,'Position']
                allele = str(combination_dataset.loc[idx,'Non_Effect_allele'].values[0])
                net_pos = int(pos) - int(start_position)
                seq_to_change = seq_to_change[:net_pos-1] + allele + seq_to_change[net_pos:]
            sequences.append(seq_to_change)
        sequences.append(self.snp_sequence)
        sequences_named = dict(zip(comb_names,sequences))
        return sequences_named
                
                
    def seq_combination(self,dataset,sign_num = 'null',window = 1000):
        self.check_ld_snps(dataset,window)
        self.obtain_snp_sequence(dataset)
        self.combination_seq = self.obtain_all_comb_seq(dataset,sign_num,window)
        return self.combination_seq
        
    
    def __str__(self):
        return "The SNP in object is "+self.rsid
        
 #DL model
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)



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

            diff = np.log2(ref_preds/(1-ref_preds)) - np.log2(alt_preds/(1-alt_preds)) 
            comb_diff_pred[comb_seq] = diff
    
    
    return comb_diff_pred       
        
def group_check(snp_feature_list,snp_comb_name_list):
    snp_groups_bool = []
    for k in snp_comb_name_list:
        snp_groups_bool.append('_' not in k)
    snp_groups_bool = sum(([ss]*2 for ss in snp_groups_bool),[])
    group1 = snp_feature_list[snp_groups_bool]

    # changed with absolute values but keep the sign
    a = np.argmax(np.abs(group1),axis=0)
    b = np.array(range(0,2002))
    group1_overall = [group1[x][y] for x, y in zip(a,b)]

 
    
    group2_idxs = [not idx for idx in snp_groups_bool]
    group2 = snp_feature_list[group2_idxs]
    group2_sub = np.subtract(group1_overall,group2)
    group2_ratio = np.divide(group2,group1_overall)

    return group1,group2,group1_overall,group2_sub,group2_ratio,group2_idxs,snp_groups_bool


def correlation_check(snp_feature_list, snp_comb_name_list):
    
    filtered_combs = []
    for snp1 in range(0,len(snp_feature_list)):
        for snp2 in range(0,len(snp_feature_list)):
            corr,_ = pearsonr(snp_feature_list[snp1],snp_feature_list[snp2])
            if -0.5 <= corr <= 0.5:
                filtered_combs.append(snp1)
                filtered_combs.append(snp2)
    filtered_combs = list(set(filtered_combs))            
    return filtered_combs        
        


## Reading initial dataset
parser = argparse.ArgumentParser()
   
parser.add_argument('-ss', '--summarystat', action='store_true', 
    help="path for summary statistics file in the format provided")
parser.add_argument('-m', '--model', action='store_true',default = 'deepsea.beluga.pth',
    help="path for deep learning model")
parser.add_argument('-f', '--features', action='store_true',default = 'deepsea_beluga_2002_features.tsv.txt',
    help="path for deep learning model features information")  
parser.add_argument('-n', '--numsnps', action='store_true', 
    help="Total number of top SNPs for processing")

#parser.add_argument('-e', '--email', action='store_true', help="Entrez login email")
#parser.add_argument('-ek', '--apikey', action='store_true', help="Entrez login api key")
#parser.add_argument('-bp', '--inputsize'action='store_true', help="DNA input sequence size in bp")
args = parser.parse_args()


## Keys for access to cloud
Entrez.email  = "pradluzog@gmail.com"
Entrez.api_key = "98ad62666b4bd2dc831f1824727d74d67c08"

## Filtering the top N snps
ss = pd.read_csv(args.summarystat, sep='\t')
ss = ss.sort_values(by = ['Pvalue'], ascending=True)
n = int(args.numspns)
top_n_snps = ss.iloc[0:n,:]

## Obtaining features
features = pd.read_csv(args.features, sep = '\t')
features['feature_names'] = features['Cell type'] +'__'+ features['Assay']+'__'+ features['Assay type']
features_ids_dnase = [features['Assay type']=='DNase']
features_ids_tf = [features['Assay type']=='TF']
features_ids_histone = [features['Assay type']=='Histone']
feature_names = features['feature_names']


## Obtaining the Hg19 Position for the SNPs
for index,row in top_n_snps.iterrows():
    response = Entrez.efetch(db='SNP', id=str(top_n_snps.loc[index,'MarkerName'])).read()
    response = response[:-1]
    response_o = xmltodict.parse(response)
    pos = response_o['DocumentSummary']['CHRPOS']
    pos = pos.split(':')[1]
    top_n_snps.loc[index,'Position'] = int(pos)

## Inputing the resources for Expect.py
inputsize = 2000
batchSize = 32
maxshift = 800
args_cuda = False

## Importing the DL Model
model = Beluga()
model.load_state_dict(torch.load(args.model))
model.eval()

## Running model for top n SNPs
igap_19 = pd.read_csv('IGAP_Chr19_Hg38loc.csv')

data_matrix_grp1 = {}
data_matrix_grp2 = {}
data_matrix_grp1_overall = {}
for k in range(0,len(top_n_snps)):
    snp_test = top_n_snps.iloc[k,2]
    if snp_test not in skip_snps:
        print("Running %s ...."%(snp_test))
        snp_obj = SNP(top_n_snps.iloc[k,2],top_n_snps.iloc[k,1],top_n_snps.iloc[k,0])
        print("Obtaining combinations for %s ...."%(snp_test))
        snp_comb_seq = snp_obj.seq_combination(igap_19)
        print("Predicting the sequence profiles for %s ...."%(snp_test))
        comb_diff_pred = get_predicted_diff(snp_comb_seq)
        f = h5py.File("Major_Minor" + snp_test +'.diff.h5', 'w')
        key_names = list(comb_diff_pred.keys())
        print("Saving the result for %s"%(snp_test))
        for i in key_names:
            f.create_dataset(i, data=comb_diff_pred[i])
        f.close()
        
        # Am I running with the threshold combination or not?
        snp_vals,comb_vals,group1,met2_sub,met2_ratio,idd,snp_groups_bool = group_check(comb_diff_pred,key_names)      
        data_matrix_grp1_overall[t_name] = {'data': group1,'ylabels':list(feature_names)}
        xlabs = key_names
        xlabs = [[x]*2 for x in xlabs]
        xlabs =  [item for sublist in xlabs for item in sublist]
        iddx = [idd[i] for i in range(0,len(idd),2)]
        xlabels = [key_names[i] for i in range(0,len(iddx)) if iddx[i]]
        xlabels = sum(([ss]*2 for ss in xlabels),[])
        data_matrix_grp2[t_name] = {'data': met2_sub,'ylabels':xlabels, 'xlabels':list(feature_names)}
        kk = [snp_groups_bool[ss] for ss in range(0,len(snp_groups_bool),2)]
        xlabels = list(compress(key_names, kk))
        xlabels = sum(([ss]*2 for ss in xlabels),[])
        data_matrix_grp1[t_name] = {'data': snp_vals,'ylabels':xlabels, 'xlabels':list(feature_names)}


with open('DataMatrixGrp1_Grp2.pickle', 'wb') as file:
    pickle.dump(data_matrix_grp2_grp1, file, protocol=pickle.HIGHEST_PROTOCOL)
with open('DataMatrixGrp1.pickle', 'wb') as file:
    pickle.dump(data_matrix_grp1, file, protocol=pickle.HIGHEST_PROTOCOL)



      
      
      

