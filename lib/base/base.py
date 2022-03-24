

import inspect
import hashlib
import gc
import requests
import collections
import importlib
import copy
import time
import string
import math
import os
import sys
import random
random.seed(0)
import shutil
import functools
import itertools
import json
import pickle
import tarfile
import warnings
import logging
import subprocess
from pathlib import Path
from pprint import pprint
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
import torchmetrics
import transformers as tfs

from tqdm import tqdm
import seaborn as sns
import matplotlib

tqdm.pandas()
VERBOSE = ['timed']
VERBOSE = []

os.environ["TOKENIZERS_PARALLELISM"] = 'false'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
pl.seed_everything(0, workers=True)


def split_list_into_equal_parts(lst,w,pad=-1):
    s_len = len(lst)
    chunks = []
    start = 0
    while start<s_len:
        chunk = lst[start:start+w]
        if len(chunk)<w:
            chunk+=[pad]*(w-len(chunk))
        start += w
        chunks.append(chunk)
    return chunks    


def show_log(m):
    df = load(Path('log',m,'version_0','metrics.csv'))
    df = df.drop(columns=['step','epoch'])
    print(df)
###

def filter_dict(d, inc=[], exc=[]):
    if inc:
        return {
            k:v for k,v in d.items() if k in inc}
    if exc:
        return {
            k:v for k,v in d.items() if k not in exc}

def download_large_file(url, save_path):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)
            
            
def info(x):
    n = type(x).__name__
    if isinstance(x,list) or isinstance(x,tuple):
        print(n, len(x))
    elif isinstance(x,torch.Tensor):
        print(n, torch.tensor(x.shape).tolist())
    elif isinstance(x,dict):
        print(n, list(x.keys()))  
    else:
        print(x)
        
def encode(s):
    return hashlib.md5(
        s.encode('utf-8')).hexdigest() 


def size_in_bytes(tensor):
    return tensor.element_size()*tensor.nelement()


def dir_size_MB(start_path):
    total_size = 0
    for dirpath,dirnames,filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size/1024/1024


def load(path): 
    path = Path(path)
    if path.suffix == '.json':
        with open(path,'r') as f: x = json.load(f)
    if path.suffix == '.csv':
        x = pd.read_csv(path)
    if path.suffix == '.feather':
        x = pd.read_feather(path)
    if path.suffix == '.pth': 
        x = torch.load(
            path, map_location = 'cuda' \
                if torch.cuda.is_available() else 'cpu')
    return x
            
            
def save(x,path): 
    path = Path(path)
    if path.suffix == '.json':
        with open(path,'w') as f: json.dump(x,f)  
    if path.suffix == '.csv':
        x.to_csv(path)
    
    
def add_to_sys_path(dir_path):
    for path in Path(dir_path).iterdir():
        sys.path.append(str(path)) 
        
#add_to_sys_path('git')


def all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            old = getattr(cls, attr)
            if callable(old):
                new = decorator(old)
                setattr(cls, attr, new)
        return cls
    return decorate


def get_set(func, *args, **kwargs):
    def _func(*args):
        assert len(args)==2
        _self, name = args[0], args[1]
        x = getattr(_self, name)
        x = func(_self, x, **kwargs)
        setattr(_self, name, x)
    return _func


def verbose(func, *args, **kwargs):
    def _func(*args, **kwargs):
        class_name = type(args[0]).__name__
        print(class_name, func.__name__+'...')
        return func(*args, **kwargs) 
    return _func 
        

def timed(func, *args, **kwargs):
    def _func(*args, **kwargs):
        try:
            class_name = type(args[0]).__name__
        except:
            class_name = ''
        print(class_name, func.__name__+'...', end=' ')
        time.sleep(0.1)
        t = time.time()
        result = func(*args, **kwargs)
        print('['+str(math.ceil(time.time()-t))+']')
        #print('['+str(time.time()-t)+']')
        return result 
    return _func if 'timed' in VERBOSE else func



     
    
def extract_from_tar_gz(tar_gz_path, dir_path):
    with tarfile.open(tar_gz_path) as f:
        for item in tqdm(f.getmembers()):
            f.extract(item, path=dir_path)                  
            




    
    
    
    
