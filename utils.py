import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import random
import pyfasta

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_weights_for_balanced_classes(dataset,nfeatures, nclasses=2):
    mat=np.power(nclasses,np.arange(nfeatures))
    label=dataset.numpy().dot(mat[:,None])
    N=dataset.size(0)
    lb,cts=np.unique(label,return_counts=True)
    weight=[]
    for i,val in enumerate(label):
        idx=np.where(lb==val)
        w=N/(cts[idx]+1)
        weight.append(w)
    weight=np.asarray(weight).flatten()
    return weight

def one_hot(labels:torch.Tensor,num_classes:int, eps = 1e-6) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}".format(type(labels)))
    n,m = labels.size(0),label.size(1)
    onehot = torch.zeros((n, num_classes))
    return onehot.scatter_(2,labels.unsqueeze(2), 1.0) + eps

class TrainData(Data.Dataset):
    def __init__(self, seq_file,label_file,root_dir):
        self.seq_data=torch.from_numpy(np.load(root_dir+seq_file))
        self.label_data=torch.from_numpy(np.load(root_dir+label_file))
        self.root_dir=root_dir
    def __len__(self):
        return len(self.seq_data)
    def __getitem__(self, idx):
        data=(self.seq_data[idx],self.label_data[idx])
        return data

def encodeSeqs(seqs, inputsize):
    seqsnp = np.zeros((len(seqs), 4, inputsize))
    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),'C': np.asarray([0, 0, 1, 0]),
            'T': np.asarray([0, 0, 0, 1]),'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),'a': np.asarray([1, 0, 0, 0]),
            'g': np.asarray([0, 1, 0, 0]), 'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]), 'n': np.asarray([0, 0, 0, 0]),
            '-': np.asarray([0, 0, 0, 0])}
    n=0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = mydict[c]
        n = n + 1
    dataflip = seqsnp[:, ::-1, ::-1]
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp


