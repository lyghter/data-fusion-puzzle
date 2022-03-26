



from ..base import *
from ..data.splitter.sequential2 import Sequential2Splitter
from ..data.datamodule.test2 import Test2


def _main():
        data_dir, pred_file =sys.argv[1].split('--')[1:]
        data_dir = Path(data_dir)
        pred_file = Path(pred_file)
        print(data_dir) #/data
        print(pred_file) #/output/predictions.npz
        print(os.listdir(data_dir)) #['clickstream.csv', 'transactions.csv']

        root = Path('json')
        event_encoder = Encoder(root,'event')
        uid_encoder = Encoder(root,'uid') 
        uid = 'user_id'
        ts = 'timestamp'

        cl = pd.read_csv(data_dir/'clickstream.csv')
        cl[ts] = cl[ts].progress_apply(pd.Timestamp)
        event = 'cat_id'
        cl[event] = cl[event].apply(
            lambda x: f'rtk_{x}')
        cl[event] = s.encoder.transform(cl[event])
        cl[uid] = s.encoder.transform(cl[uid])

        tr = pd.read_csv(
            data_dir/'transactions.csv')
        tr = tr.rename(
            columns={'transaction_dttm': ts})
        tr[ts] = tr[ts].progress_apply(pd.Timestamp)
        event = 'mcc_code'
        tr[event] = tr[event].apply(
            lambda x: f'bank_{x}')
        tr[event] = s.encoder.transform(tr[event])
        tr[uid] = s.encoder.transform(tr[uid]) 

        splitter = Sequential2Splitter()
        XC,YC,XT,YT = splitter.run(cl,tr)

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
        a.bank_len = XT.shape[1]
        a.rtk_len = XC.shape[1] 
        
        bank = sorted(
            tr.user_id.unique().tolist())
        rtk = sorted(
            cl.user_id.unique().tolist())   
        max_len = max(len(bank),len(rtk))
        df = pd.DataFrame(index=range(max_len))
        df['bank'] = bank
        df['rtk'] = rtk
        df = df.fillna(-1)
        
        def collate(DD):
            AB = [A+B for A in 'XY' for B in 'TC']
            kk = AB+['MT','MC']+['bank','rtk','M']
            B = {k:[] for k in kk}
            for D in DD:
                for k in B:
                    if k in D:
                        B[k].append(D[k])
            for k in B:
                if k in AB+['MT','MC']:
                    B[k] = torch.cat(B[k])
                if k in ['bank','rtk','M']:
                    B[k] = torch.tensor(B[k])
            return B
        
        c = Args()
        c.event_encoder = event_encoder
        c.test = Test2(df, collate, a)
        
        model = Model(a,c)
        
        callbacks = [
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                save_weights_only = bool(0),
                filename = '{R1} {MRR} {P}', 
                monitor = 'R1', 
                verbose = False,
                save_last = bool(1),
                save_top_k = 1, 
                mode = 'max', 
            ),
        ]
        trainer = pl.Trainer(
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
        
        p = s.a.log_dir
        p /= s.a.exp_name
        p /= 'version_0'
        p /= 'checkpoints'
        a.ckpt = p/'last.ckpt'
        print(a.ckpt.stem)
        BB = trainer.predict(
            model, c.test, ckpt_path=a.ckpt)  
        pred = model.predict_epoch_end(BB)
        print(pred.sample().T)
        
        path = a.csv_dir/f'{a.ckpt.stem}.csv'
        c = 'rtk_list'
        pred[c] = pred[c]\
            .apply(lambda x: str(x))\
            .replace("'", '', regex=True)
        pred.to_csv(path, index=False)



    print(cl.user_id.unique()),'[000143baebad4467a23b98c918ccda19]'
    print(cl.timestamp.min()),'2021-01-30 20:08:12' 
    
    t0 = time.time()    
    tr = pd.read_csv(data_dir/'transactions.csv')
    t1 = time.time() 
    print(t1-t0), '0.002720355987548828'
    print(tr.shape),'(40, 5)'
    print(tr.user_id.unique()),['4e2ede603cbd41dc9155ac919818ab6d' '518dccd73b844e18b67a71f2dae925cd']
    print(tr.transaction_dttm.min()),'2020-08-01 20:39:00'
    





    
if __name__=='__main__':
    main()


