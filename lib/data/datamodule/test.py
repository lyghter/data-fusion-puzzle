



from ...base import *
from ..dataset import DataSet
from ..io import IO


class Test(pl.LightningDataModule, IO):
    def __init__(s, c):
        super().__init__() 
        s.name = 'TEST'
        s.c = c
        s.data_dir = c.a.data_dir
        s.a = c.a
        
        
    def prepare_data(s):
        super().prepare_data()
        s.P = s.load(s.name+'.feather').sample(
            frac=s.a.test_frac, random_state=0)
        print(f'P: {len(s.P)}')
        s.ds = {'P': DataSet(s,'P')}
           
        
    def predict_dataloader(s):
        s.N = 'P'
        return DataLoader(
            dataset = s.ds['P'],
            batch_size = s.a.val_batch_size,
            pin_memory = True,
            num_workers = s.a.num_workers,
            collate_fn = s.c.collate,    
            shuffle = False,
            drop_last = False,  
        )  
