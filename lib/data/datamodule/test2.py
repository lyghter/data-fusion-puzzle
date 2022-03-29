



from ...base import *
from ..dataset2 import DataSet2
from ..io import IO


class Test2(pl.LightningDataModule, IO):
    def __init__(s, d, collate, a):
        super().__init__() 
        s.d = d
        s.P = d.P
        s.collate = collate
        s.a = a
        
        
    def prepare_data(s):
        super().prepare_data()
        print(f'P: {len(s.d.P)}')
        s.ds = {'P': DataSet2(s.d, s.a)}
           
        
    def predict_dataloader(s):
        s.N = 'P'
        return DataLoader(
            dataset = s.ds['P'],
            batch_size = s.a.val_batch_size,
            pin_memory = True,
            num_workers = s.a.num_workers,
            collate_fn = s.collate,    
            shuffle = False,
            drop_last = False,  
        )  
