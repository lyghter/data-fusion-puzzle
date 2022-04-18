



from ..base import *
from ..data.downloader import Downloader
from ..data.encoder.event import EventEncoder
from ..data.encoder.uid import UidEncoder
from ..data.splitter.sequential import SequentialSplitter
from ..data.splitter.sequential2 import Sequential2Splitter
from ..data.prepare.preprocessor import Preprocessor
from ..data.prepare.vectorizer import Vectorizer
from ..data.datamodule.train import Train
from ..data.datamodule.test import Test
from .model import Model
from ..data.io import IO


    
class Estimator(IO):     
    def __init__(s,a):
        s.AB = [A+B for A in 'XY' for B in 'TC']
        s.a = a
        s.data_dir = a.data_dir
        s.data_dir.mkdir(exist_ok=True)
        s.data_files = set(os.listdir(s.data_dir))
        s.load_encoders()
        s.trainer = s.get_trainer(s.a)
        

    def run_or_pass(s, class_name, files):
        Class = eval(class_name)
        t0 = time.time()
        if not files.issubset(s.data_files): 
            Class(s).run()
        t1 = time.time()
        dt = round(t1-t0)
        print(f'{class_name}: {dt} s')
        print('-'*20)
    
        
    def load_encoders(s):          
        for name in ['event','uid']:
            e = Encoder(Path('json'),name)
            e.load()
            setattr(s, name+'_encoder', e)    
            
                       
    def update_args(s):
        s.a.bank_len = s.XT.shape[1]
        s.a.rtk_len = s.XC.shape[1]  
        print('bank_len', s.a.bank_len)
        print('rtk_len', s.a.rtk_len)  
    
    
    def collate(s, DD):
        B = {k:[] for k in DD[0].keys()}
        for D in DD:
            for k in B:
                if k in D and D[k] is not None:
                    B[k].append(D[k])
        for k in B:
            if k in s.AB+['MT','MC']:
                try:
                    B[k] = torch.cat(B[k])
                except:
                    pass
            if k in ['bank','rtk','M','uid']:
                B[k] = torch.tensor(B[k])
        return B
    
        
    def fit(s):
        a = s.a
        if not a.ckpt:
            shutil.rmtree(
                a.log_dir/a.exp_name,
                ignore_errors=True)
        s.run_or_pass(
            'Downloader', {
#             'train_matching.csv',
#             'mcc_codes.csv',
#             'click_categories.csv',
#             'currency_rk.csv',
#             'puzzle.csv',
#             'sample_submission.csv',
            'transactions.feather',
        })
        s.run_or_pass(
            'EventEncoder', {
            'bank.feather',
            'rtk.feather',
#             'transactions_events.feather',
#             'clickstreams_events.feather',
        })
        s.run_or_pass(
            'UidEncoder', {
            'TRAIN.feather',
            'TEST.feather',
#             'transactions_events_uids.feather',
#             'clickstreams_events_uids.feather',
        })  
        s.run_or_pass(
            a.splitter+'Splitter', {
            'XC.pt','YC.pt','XT.pt','YT.pt',
        })          
        for k in s.AB:
            setattr(s, k, s.load(f'{k}.pt'))
        s.update_args()
        s.model = Model(s)
        s.train = Train(s)
        s.trainer.fit(
            s.model, s.train, ckpt_path=a.ckpt)
        
        
    def predict_puzzle(s, stem_ext):
        s.run_or_pass(
            'Preprocessor', {
                'cl_P.feather',
                'tr_P.feather',
        })
        s.a.cl_path = s.a.data_dir/'cl_P.feather' #clickstream.csv
        s.a.tr_path = s.a.data_dir/'tr_P.feather' #transactions.csv
        s.a.L = 'P' 
        pred = s.predict(stem_ext)
        print(pred.sample().T)
        s.save_prediction_csv(pred)        

        
    def predict_matching(s, stem_ext):
        s.a.cl_path = s.a.data_dir/'cl.csv' 
        s.a.tr_path = s.a.data_dir/'tr.csv' 
        
        s.a.cl_path = s.a.data_dir/'clickstream.csv' 
        s.a.tr_path=s.a.data_dir/'transactions.csv' 

        s.a.L = 'M' 
        pred = s.predict(stem_ext)
        print(pred.sample().T)
        s.save_prediction_npz(pred)
        
        
    def predict(s, stem_ext):
        L = s.a.L
        s.run_or_pass(
            'Vectorizer', {
                f'XC{L}.pt',
                f'YC{L}.pt',
                f'XT{L}.pt',
                f'YT{L}.pt'
        })
        for k in s.AB:
            setattr(s, k, s.load(f'{k}{L}.pt'))
        s.test = Test(s)   
        p = s.a.log_dir
        p /= s.a.exp_name
        p /= 'version_0'
        p /= 'checkpoints'
        s.a.ckpt = p/stem_ext
        s.update_args()
        s.model = Model(s)
        BB = s.trainer.predict(
            s.model, s.test, ckpt_path=s.a.ckpt)  
        pred = s.model.predict_epoch_end(BB)
        return pred
        
        
    def save_prediction_csv(s, pred):
        path = s.a.csv_dir/f'{s.a.ckpt.stem}.csv'
        c = 'rtk_list'
        pred[c] = pred[c]\
            .apply(lambda x: str(x))\
            .replace("'", '', regex=True)
        pred.to_csv(path, index=False)
        
        
    def save_prediction_npz(s, pred):
        c = 'rtk_list'
        pred[c] = pred[c].apply(
            lambda x: ([0.0, 0]+x)[:100])
        print(pred.values)
        np.savez(str(s.a.pred_file), pred.values)
        
        
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
