


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
        kk = ['XT','YT','XC','YC','bank','rtk']
        D = {k:None for k in kk}
        D['uid'], D['label'] = s.labeled_uids[i]
        if D['label']=='bank':
            D['XT'] = s.e.XT[s.e.YT==D['uid']].int()
            D['YT'] = s.e.YT[s.e.YT==D['uid']].int()
        if D['label']=='rtk':
            D['XC'] = s.e.XC[s.e.YC==D['uid']].int()
            D['YC'] = s.e.YC[s.e.YC==D['uid']].int()
        return D

        