



from ...base import *
from ..io import IO


@all_methods(verbose)
class SequentialSplitter(IO):
    def __init__(s, a):
        s.data_dir = a.data_dir
        s.pp = a.splitter_pp
        s.c2l = {}
     
    
    def run(s):
        s.get_tensors(
            'clickstreams_events_uids',
            'cat_id',
            'rtk'
        )
        s.get_tensors(
            'transactions_events_uids',
            'mcc_code',
            'bank'
        )  


    def get_tensors(s, df_name, feat_name, id_name):
        
        dd = []
        DF = s.load(f'{df_name}.feather')
        for uid,df in tqdm(DF.groupby('user_id')):
            c = 'timestamp'
            t0 = pd.Timestamp(str(df[c].min())[:7])
            t1 = pd.Timestamp(str(df[c].max())[:7]) 
            t = t0
            w = pd.Timedelta(
                days=s.pp['n_days_in_sample'])
            while t<=t1:
                x = df[(df[c]>=t)&(df[c]<=t+w)]
                t += w
                l = x[feat_name].tolist()
                if not len(l):
                    l.append(0)
                dd.append({'uid':uid,'list':l})
        
        df = pd.DataFrame(dd)
        s.c2l[id_name] = s.get_max_len(df,id_name)
        s.pad(df, id_name)
        s.X = torch.tensor(df['list'].tolist()) 
        s.Y = torch.tensor(df['uid'].tolist())
        print(s.X.shape)
        print(s.Y.shape)
        uids = set(s.Y.tolist())
        print(f'{id_name}_uids',len(uids))
        B = df_name[0].upper() 
        for A in 'XY':
            s.save(A, f'{A}{B}.pt')
            
        
    def get_max_len(s, df, id_name):
        q = s.pp[f'{id_name}_quantile']
        c = 'list'
        x = df[df[c].apply(len)>1]
        return int(x[c].apply(len).quantile(q))

        
    def pad(s, df, id_name):
        c = 'list'    
        max_len = s.c2l[id_name]
        df[c] = df[c]\
            .progress_apply(
                lambda l: l[:max_len])\
            .progress_apply(
                lambda l: l+[0]*(max_len-len(l)))
        assert df[c].apply(len).nunique()==1
   
            
 