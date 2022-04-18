





from ...base import *
from ..io import IO
from ..splitter.sequential2 import Sequential2Splitter


class Preprocessor(IO):
    def __init__(s, e):
        a = e.a
        s.a = a
        s.data_dir = a.data_dir
        
        
    def run(s):
        root = Path('json')
        event_encoder = Encoder(root,'event')
        event_encoder.load()
        uid_encoder = Encoder(root,'uid') 
        uid_encoder.load()
        
        s.df = s.load('TEST.feather')  
        s.df['bank']=uid_encoder.inverse_transform(
            s.df['bank'])
        s.df['rtk']=uid_encoder.inverse_transform(
            s.df['rtk'])
        s.cl = s.load('_clickstreams.feather')
        s.tr = s.load('_transactions.feather')
        s.cl_P=s.cl[
            s.cl.user_id.isin(s.df.rtk.tolist())]
        s.tr_P=s.tr[
            s.tr.user_id.isin(s.df.bank.tolist())]
        
        s.remove('_clickstreams.feather')
        s.remove('_transactions.feather')        
        
        s.save('cl_P','cl_P.feather')
        s.save('tr_P','tr_P.feather')   
        