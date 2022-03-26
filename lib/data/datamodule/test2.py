



from ...base import *
from ..dataset import DataSet
from ..io import IO


class Test(pl.LightningDataModule, IO):
    def __init__(s, P, collate, a):
        super().__init__() 
        s.P = P
        s.collate = collate
        s.a = a
        
        
    def prepare_data(s):
        super().prepare_data()
        print(f'P: {len(s.P)}')
        s.ds = {'P': DataSet(s,'P')}
           
        
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
