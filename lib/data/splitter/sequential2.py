



from ...base import *
from ..io import IO


@all_methods(verbose)
class Sequential2Splitter(IO):
    def __init__(s):
        s.pp = dict(
            n_days_in_sample = 30,
            bank_quantile = 0.9,
            rtk_quantile = 0.9,
        )
        s.c2l = {'bank': 120, 'rtk': 2256}
     
    
    def run(s,cl,tr):
        XC,YC = s.get_tensors(cl,'cat_id','rtk')
        XT,YT = s.get_tensors(tr,'mcc_code','bank')
        return XC,YC,XT,YT 


    def get_tensors(s, DF, feat_name, id_name):
        
        dd = []
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
        s.pad(df, id_name)
        X = torch.tensor(df['list'].tolist()) 
        Y = torch.tensor(df['uid'].tolist())
        print(X.shape)
        print(Y.shape)
        uids = set(Y.tolist())
        print(f'{id_name}_uids',len(uids))
        return X,Y
            
        
    def pad(s, df, id_name):
        c = 'list'    
        max_len = s.c2l[id_name]
        df[c] = df[c]\
            .progress_apply(
                lambda l: l[:max_len])\
            .progress_apply(
                lambda l: l+[0]*(max_len-len(l)))
        assert df[c].apply(len).nunique()==1
   
            
 