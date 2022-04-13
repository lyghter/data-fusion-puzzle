

from ..base import *
from .io import IO


class Preprocessor(IO):
    def __init__(s, a):
        s.a = a
        
        
    def run(s):
        root = Path('json')
        event_encoder = Encoder(root,'event')
        event_encoder.load()
        uid_encoder = Encoder(root,'uid') 
        uid_encoder.load()
        uid = 'user_id'
        ts = 'timestamp'
        
        try:
            cl_path = s.a.data_dir/'clickstream.csv'
            cl = pd.read_csv(cl_path)
        except:
            cl_path = s.a.data_dir/'cl.csv'
            cl = pd.read_csv(cl_path)
        
        print('len(cl)', len(cl))

        cl[ts] = cl[ts].progress_apply(pd.Timestamp)
        event = 'cat_id'
        cl[event] = cl[event].apply(
            lambda x: f'rtk_{x}')
        cl[event]=event_encoder.transform(cl[event])
        cl[uid] = uid_encoder.transform(cl[uid])
        
        try:
            tr_path= s.a.data_dir/'transactions.csv'
            tr = pd.read_csv(tr_path)
        except:
            tr_path = s.a.data_dir/'tr.csv'
            tr = pd.read_csv(tr_path)     
            
        tr = tr.rename(
            columns={'transaction_dttm': ts})
        tr[ts] = tr[ts].progress_apply(pd.Timestamp)
        event = 'mcc_code'
        tr[event] = tr[event].apply(
            lambda x: f'bank_{x}')
        tr[event]=event_encoder.transform(tr[event])
        tr[uid] = uid_encoder.transform(tr[uid]) 

        bank = sorted(
            tr.user_id.unique().tolist())
        rtk = sorted(
            cl.user_id.unique().tolist())   
        max_len = max(len(bank),len(rtk))
        df = pd.DataFrame(index=range(max_len))
        
        splitter = Sequential2Splitter()
        XC,YC,XT,YT = splitter.run(cl,tr)
        for name in ['XC','YC','XT','YT']:
            s.save(name, f'{name}P.pt')
            
        P = pd.DataFrame()
        P['bank'] = (bank+bank)[:max_len]
        P['rtk'] = (rtk+rtk)[:max_len]
        s.P = P.fillna(-1)
        s.save('P','P.feather')
        

            
