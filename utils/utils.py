import time, os
from tqdm import tqdm
import json
import pickle

import torch.nn as nn
import torch

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_json(myjson, filepath):
    jstring = json.dumps(myjson)
    jfile = open(filepath, "w")
    jfile.write(jstring)
    jfile.close()

def load_object(filepath):
    loaded = {}
    with open(filepath, "rb") as file:
        loaded = pickle.load(file)
    return loaded

def save_object(dict, filepath):
    pickle.dump( dict, open( filepath, "wb" ) )
    
    
def cosinus_distance(a, b):

    if len(a.shape) == 1: # 1D
        a = torch.unsqueeze(a,0)
    if len(b.shape) == 1: # 1D
        b = torch.unsqueeze(b,0)

    if a.shape[0] == 1:
        a = torch.transpose(a, 0, 1)
        b = torch.transpose(b, 0, 1)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    output = cos(a, b)
    #print(a.shape, b.shape)
    return 1.0-output


def euclidean_distance(a, b):
    if len(a.shape) == 1: # 1D
        a = torch.unsqueeze(a,0)
    if len(b.shape) == 1: # 1D
        b = torch.unsqueeze(b,0)

    if a.shape[0] == 1:
        a = torch.transpose(a, 0, 1)
        b = torch.transpose(b, 0, 1)

    dist = sum((a-b)**2)
    return dist

def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device
