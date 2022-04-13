



from ...base import *
from ..dataset.train import TrainDataset
from ..io import IO



class Train(pl.LightningDataModule, IO):
    def __init__(s, e):
        super().__init__()
        s.name = 'TRAIN'
        s.e = e
        s.data_dir = e.a.data_dir
        s.a = e.a

        
    def prepare_data(s):
        super().prepare_data()
        s.fit_val_split()
        s.build_ds()
            

    def fit_val_split(s):
        df = s.load(s.name+'.feather').sample(
            frac=s.a.train_frac, random_state=0)
        kf = KFold(
            random_state = s.a.random_state,
            n_splits = s.a.n_folds, 
            shuffle = True, 
        )        
        ff = [v for f,v in kf.split(df)] 
        val = ff.pop(s.a.fold)
        fit = np.concatenate(ff)
        s.V = df.iloc[val]
        s.F = df.iloc[fit]
        print(f'F: {len(s.F)}')
        print(f'V: {len(s.V)}')  
         
     
    def build_ds(s):
        s.ds = {N: TrainDataset(s, N) for N in 'FV'}

    
    def train_dataloader(s):
        s.N = 'F'
        return DataLoader(
            dataset = s.ds['F'],
            batch_size = s.a.fit_batch_size,
            pin_memory = True,
            num_workers = s.a.num_workers,
            collate_fn = s.e.collate,    
            shuffle = True,
            drop_last = True,  
        )  
            
            
    def val_dataloader(s):
        s.N = 'V'
        return DataLoader(
            dataset = s.ds['V'],
            batch_size = s.a.val_batch_size,
            pin_memory = True,
            num_workers = s.a.num_workers,
            collate_fn = s.e.collate,    
            shuffle = False,
            drop_last = False,  
        )  
        

      