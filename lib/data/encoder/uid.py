


from ...base import *
from ..io import IO


@all_methods(verbose)
class UidEncoder(IO):
    def __init__(s, a):
        s.data_dir = a.data_dir
        
        
    def run(s):
        s.prepare()
        s.transform_train()
        s.transform_test()  

        
    def prepare(s): 
        c = 'user_id'
        tr_name = 'transactions_events.feather'
        cl_name = 'clickstreams_events.feather'
        s.tr = s.load(tr_name)
        s.cl = s.load(cl_name)
        tr = s.tr[c].unique().tolist()
        cl = s.cl[c].unique().tolist()
        vocab = sorted(set(tr+cl))
        s.encoder = Encoder(Path('json'),'uid')
        s.encoder.fit(vocab)
        s.tr[c] = s.encoder.transform(s.tr[c])
        s.cl[c] = s.encoder.transform(s.cl[c])
        s.save('tr',tr_name.replace('.','_uids.'))
        s.save('cl',cl_name.replace('.','_uids.')) 
        s.encoder.save()
            
        
    def transform_train(s): 
        df = s.load('train_matching.csv')
        df = df[df['rtk']!='0']
        assert df.bank.nunique()==df.rtk.nunique()
        for c in ['bank','rtk']:
            df[c] = s.encoder.transform(df[c])
        s.df = df
        s.save('df','TRAIN.feather')
        
        
    def transform_test(s):
        df = s.load('puzzle.csv')
        for c in ['bank','rtk']:
            df[c] = s.encoder.transform(df[c])
        s.df = df
        s.save('df','TEST.feather')

