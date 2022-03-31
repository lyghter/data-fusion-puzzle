


from ...base import *


class TestDataset(Dataset):
    def __init__(s, d):
        s.d = d
        bank = sorted(set(d.YT.tolist()))
        rtk = sorted(set(d.YC.tolist()))
        bank = [[uid,'bank'] for uid in bank]
        rtk = [[uid,'rtk'] for uid in rtk]
        s.labeled_uids = bank+rtk

    def __len__(s):
        return len(s.labeled_uids)     
    
    def __getitem__(s, i): 
        uid, label = s.labeled_uids[i]
        if label=='bank':
            X = s.d.XT[s.d.YT==uid]
        if label=='rtk':
            X = s.d.XC[s.d.YC==uid]
        return dict(uids=uid, X=X, labels=label)

        
# XT,YT,XC,YC        