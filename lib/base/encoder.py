


from .base import *
from . import io 


class Encoder:
    def __init__(s, dir_path, name):
        s.dir_path = dir_path
        s.name = name

    def fit(s, vocab):
        s.v2i = dict(zip(vocab,range(len(vocab))))
        s.i2v = dict(zip(range(len(vocab)),vocab)) 

    def load(s):
        s.v2i = io.load(
            s.dir_path/f'{s.name}_v2i.json')
        i2v = io.load(
            s.dir_path/f'{s.name}_i2v.json')
        s.i2v = {
            int(i):v for i,v in i2v.items()}
    
    def save(s):
        io.save(
            s.v2i, s.dir_path/f'{s.name}_v2i.json')
        io.save(
            s.i2v, s.dir_path/f'{s.name}_i2v.json')
        
    def transform(s, vv):
        return [s.v2i[v] for v in vv]
    
    def inverse_transform(s, ii):
        return [s.i2v[i] for i in ii] 
    
    def __len__(s):
        return len(s.v2i)


