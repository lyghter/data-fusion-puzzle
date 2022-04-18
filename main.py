



from lib.base import *
from lib.data.splitter.sequential2 import Sequential2Splitter
from lib.run.args import Args
from lib.data.datamodule.test import Test
from lib.run.model import Model
from lib.data.io import IO
from lib.run.estimator import Estimator

warnings.filterwarnings('ignore')
    
    
def main():
        a = Args(
            splitter = 'Sequential',
            splitter_pp = dict(
                n_days_in_sample = 30,
                bank_quantile = 0.9,
                rtk_quantile = 0.9,
            ),    
            n_folds = 3,# 1000 == 'full train'
            fold = 0,

            fit_limit = 1.,
            val_limit = 1.,

            batch_size = 32,    
            lr = 2e-3,
            n_epochs = 10,
            check_val_every_n_epoch = 1,

            bb_pp = dict(
                block_size = 16,
                hidden_size = 128,
                intermediate_size = 128,
                num_attention_heads = 1,
                num_hidden_layers = 1,
                num_random_blocks = 1,
            ),

            loss = 'MarginLoss',
            loss_pp = dict(),

            use_unmatched = bool(0),

            miner = None,
            miner_pp = dict(),

            avg_loss = 'mean',
            avg_pred = 'mean',
        )
        a.update()
        
#         a.data_dir, a.pred_file = sys.argv[1].split('--')[1:]
        
        a.data_dir, a.pred_file= 'data','pred.npz'   
        
        a.data_dir = Path(a.data_dir)
        a.pred_file = Path(a.pred_file)
        print(a.data_dir) #/data
        print(a.pred_file) #/output/predictions.npz
        print(os.listdir(a.data_dir)) #['clickstream.csv', 'transactions.csv']        
        
        
#         a.bank_len = XT.shape[1]
#         a.rtk_len = XC.shape[1] 
        a.docker = bool(1)
        
        e = Estimator(a) 
        e.predict_matching('last.ckpt')
        
        
if __name__=='__main__':
    main()



