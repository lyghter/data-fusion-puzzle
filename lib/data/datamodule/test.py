



from ...base import *
from ..dataset.test import TestDataset
from ..io import IO


class Test(pl.LightningDataModule, IO):
    def __init__(s, e):
        super().__init__() 
        s.name = 'TEST'
        s.e = e
        s.data_dir = e.a.data_dir
        s.a = c.a

        
    def prepare_data(s):
        super().prepare_data()
        print(f'P: {len(s.P)}')
        s.ds = {'P': TestDataset(s)}
           
        
    def predict_dataloader(s):
        s.N = 'P'
        return DataLoader(
            dataset = s.ds['P'],
            batch_size = s.a.val_batch_size,
            pin_memory = True,
            num_workers = s.a.num_workers,
            collate_fn = s.e.collate,    
            shuffle = False,
            drop_last = False,  
        )  
