


from ..base import *
from ..data.downloader import Downloader
from ..data.encoder.event import EventEncoder
from ..data.encoder.uid import UidEncoder
from ..data.splitter.sequential import SequentialSplitter
from ..data.preprocessor import Preprocessor
from ..data.datamodule.train import Train
from ..data.datamodule.test import Test
from .model import Model
from ..data.io import IO


    
class Estimator(IO):     
    def __init__(s,a):
        s.a = a
        s.data_dir = a.data_dir
        s.data_dir.mkdir(exist_ok=True)
        s.data_files = set(os.listdir(s.data_dir))
        s.prepare_data_for_puzzle()
        s.prepare_data_for_matching()
        s.load_encoders()
        s.load_tensors()
        s.update_args()
        s.model = Model(a,s)
        s.trainer = s.get_trainer(s.a)
        
            
    def prepare_data_for_puzzle(s):
        a = s.a
        s.run_or_pass(
            Downloader, {
            'train_matching.csv',
            'mcc_codes.csv',
            'click_categories.csv',
            'currency_rk.csv',
            'puzzle.csv',
            'sample_submission.csv',
            'transactions.feather',
        })
        s.run_or_pass(
            EventEncoder, {
            'bank.feather',
            'rtk.feather',
#             'transactions_events.feather',
#             'clickstreams_events.feather',
        })
        s.run_or_pass(
            UidEncoder, {
            'TRAIN.feather',
            'TEST.feather',
#             'transactions_events_uids.feather',
#             'clickstreams_events_uids.feather',
        })  
        s.run_or_pass(
            eval(a.splitter+'Splitter'), {
            'XC.pt','YC.pt','XT.pt','YT.pt',
        })          
     
        
    def prepare_data_for_matching(s):
        s.run_or_pass(
             Preprocessor, {
            'XCP.pt','YCP.pt','XTP.pt','YTP.pt'
        })


    def run_or_pass(s, Class, files):
        t0 = time.time()
        if not files.issubset(s.data_files):  
            Class(s.a).run()
        t1 = time.time()
        print('-'*20)
        print(t1-t0)
        print('-'*20)
    
        
    def load_encoders(s):          
        for name in ['event','uid']:
            e = Encoder(Path('json'),name)
            e.load()
            setattr(s, name+'_encoder', e)    
            
            
    def load_tensors(s):
        s.AB = [A+B for A in 'XY' for B in 'TC']
        for k in s.AB:
            path = f'{k}P.pt' if s.a.docker else f'{k}.pt'
            setattr(s, k, s.load(path))
            
            
    def update_args(s):
        s.a.bank_len = s.XT.shape[1]
        s.a.rtk_len = s.XC.shape[1]  
        print('bank_len', s.a.bank_len)
        print('rtk_len', s.a.rtk_len)  
    
    
    def collate(s, DD):
        kk = s.AB+['MT','MC']+['bank','rtk','M']
        B = {k:[] for k in kk}
        for D in DD:
            for k in B:
                if k in D:
                    B[k].append(D[k])
        for k in B:
            if k in s.AB+['MT','MC']:
                B[k] = torch.cat(B[k])
            if k in ['bank','rtk','M']:
                B[k] = torch.tensor(B[k])
        return B
    
        
    def fit(s):
        if not s.a.ckpt:
            shutil.rmtree(
                s.a.log_dir/s.a.exp_name,
                ignore_errors=True)
        s.train = Train(s)
        s.trainer.fit(
            s.model, s.train, ckpt_path=s.a.ckpt)
        
        
    def predict_puzzle(s):
        s.a.docker = False
        s.predict()

        
    def predict_matching(s):
        s.a.docker = True
        s.predict()        
        
        
    def predict(s):
        p = s.a.log_dir
        p /= s.a.exp_name
        p /= 'version_0'
        p /= 'checkpoints'
        #s.a.ckpt = list(p.iterdir())[0]
#         s.a.ckpt = p/'last.ckpt'
        s.a.ckpt = p/'epoch=7.ckpt'
        print(s.a.ckpt.stem)
        s.test = Test(s)
        BB = s.trainer.predict(
            s.model, s.test, ckpt_path=s.a.ckpt)  
        pred = s.model.predict_epoch_end(BB)
        s.save_prediction(pred)
        
        
    def save_prediction(s, pred):
        print(pred.sample().T)
        path = s.a.csv_dir/f'{s.a.ckpt.stem}.csv'
        c = 'rtk_list'
        pred[c] = pred[c]\
            .apply(lambda x: str(x))\
            .replace("'", '', regex=True)
        pred.to_csv(path, index=False)
        
    
    def get_trainer(s,a):
        callbacks = [
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                save_weights_only = bool(0),
#                 filename = '{R1} {MRR} {P}', 
                filename = '{epoch}', 
                monitor = 'R1', 
                verbose = False,
                save_last = bool(0),
                save_top_k = -1, 
                mode = 'max', 
            ),
        ]
        return pl.Trainer(
            accumulate_grad_batches = a.acc_batches,
#            val_check_interval=a.val_check_interval,
            check_val_every_n_epoch=a.check_val_every_n_epoch,
            num_sanity_val_steps = 0,
            deterministic = bool(0) if s.a.avg_loss=='median' or s.a.avg_pred=='median' else bool(1),
            benchmark = bool(1),
            gpus = a.gpus,
            precision = a.precision,
            logger = pl.loggers.CSVLogger(
                str(s.a.log_dir), name=a.exp_name),
            callbacks = callbacks,
            max_epochs = a.n_epochs,
            limit_train_batches = a.fit_limit,
            limit_val_batches = a.val_limit,
        )   
