



from ...base import *
from ..io import IO


@all_methods(verbose)
class EventEncoder(IO):
    def __init__(s, a):
        s.data_dir = a.data_dir
        
        
    def run(s):
        s.prepare()
        s.transform(
            'transactions','bank','mcc_code')  
        s.transform(
            'clickstreams','rtk','cat_id')
        
        
    def prepare(s):
        rtk_vocab = s.get_vocab(
            'click_categories',
            'rtk',
            'cat_id',
        )
        bank_vocab = s.get_vocab(
            'mcc_codes',
            'bank',
            'MCC',
        )
        special_tokens = ['0:PAD','1:EOS','2:BOS']
        vocab = sorted(
            special_tokens + 
            rtk_vocab + 
            bank_vocab
        )
        s.encoder = Encoder(Path('json'),'event')
        s.encoder.fit(vocab)
        s.bank['MCC'] = s.encoder.transform(
            s.bank['MCC']) 
        s.rtk['cat_id'] = s.encoder.transform(
            s.rtk['cat_id'])  
        s.save('bank','bank.feather')
        s.save('rtk','rtk.feather') 
        s.encoder.save()
        
    
    def get_vocab(
        s, file_name, id_name, feat_name
    ):
        c = feat_name
        x = pd.DataFrame(index=[-1])
        x[c] = [-1]
        df= s.load(f'{file_name}.csv')
        df = pd.concat([x,df])
        df[c] = df[c].apply(
            lambda x: f'{id_name}_{x}')
        vocab = df[c].unique().tolist() 
        setattr(s, id_name, df)
        return vocab
    
    
    def transform(s, df_name, id_name, feat_name):
        c = feat_name  
        s.df = getattr(s, f'load_{df_name}')()
        s.df[c] = s.df[c].apply(
            lambda x: f'{id_name}_{x}')
        s.df[c] = s.encoder.transform(s.df[c])
        name = f'{df_name}_events.feather'
        name = f'{df_name}.feather' ###
        s.save('df', name)  
        
        
    def load_clickstreams(s):
        pp = []
        for p in s.data_dir.iterdir():
            if p.suffix=='.json' and p.stem.isdigit():
                p = str(p).replace('json','feather')
                pp.append(p) 
        dfs = []
        for p in tqdm(pp):
            dfs.append(pd.read_feather(p))
            os.remove(p) ###
        return pd.concat(dfs)
    
    
    def load_transactions(s):
        return s.load('transactions.feather')
    
        