



from ..base import *


class Args:
    def __init__(a, **kwargs):
        for k,v in kwargs.items(): setattr(a,k,v)

            
    def update(a):
        a.exp_name = a.get_name()
        print(a.exp_name)
        a.gpus = 1
        a.precision = 16
        a.ckpt = None
        a.num_workers = os.cpu_count()
        a.data_dir = Path('data')
        a.log_dir = Path('log')
        a.csv_dir = Path('csv')        
        a.random_state = 0
        a.train_frac = 1.0
        a.test_frac = 1.0
        a.k = 100
        a.acc_batches = 1
        a.fit_batch_size = a.batch_size
        a.val_batch_size = a.batch_size
        a.pred_batch_size = 64*2
 
        
    def get_name(a):
        x = str(a.__dict__).encode('utf-8')
        x = hashlib.md5(x).hexdigest() 
        return x
    
    
