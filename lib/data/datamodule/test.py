



from ...base import *
from ..dataset.test import TestDataset
from ..io import IO


class Test(pl.LightningDataModule, IO):
    def __init__(s, e):
        super().__init__() 
        s.name = e.a.L
        s.e = e
        s.data_dir = e.a.data_dir
        s.a = e.a

        
    def prepare_data(s):
        super().prepare_data()
#         s.P = s.load(s.name+'.feather').sample(
#             frac=s.a.test_frac, random_state=0)
#         print(f'P: {len(s.P)}')
        s.ds = TestDataset(s)
        
#         {'P': TestDataset(s)}
        
           
        
    def predict_dataloader(s):
        s.N = 'P'
        return DataLoader(
            dataset = TestDataset(s),
            batch_size = s.a.pred_batch_size,
            pin_memory = True,
            num_workers = s.a.num_workers,
            collate_fn = s.e.collate,    
            shuffle = True,
            drop_last = False,  
        )  
