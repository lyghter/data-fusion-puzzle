


from ...base import *


class TestDataset(Dataset):
    def __init__(s, dm):
        s.e = dm.e
        bank = sorted(set(s.e.YT.tolist()))
        rtk = sorted(set(s.e.YC.tolist()))
        bank = [[uid,'bank'] for uid in bank]
        rtk = [[uid,'rtk'] for uid in rtk]
        s.labeled_uids = bank+rtk

    def __len__(s):
        return len(s.labeled_uids)     
    
    def __getitem__(s, i): 
        uid, label = s.labeled_uids[i]
        if label=='bank':
            X = s.e.XT[s.e.YT==uid]
        if label=='rtk':
            X = s.e.XC[s.e.YC==uid]
        return dict(uids=uid, X=X, labels=label)

        