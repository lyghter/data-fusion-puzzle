

from ..base import *

   
class DataSet(Dataset):
    def __init__(s, dm, N):
        s.c = dm.c
        s.N = N
        s.df = getattr(dm,s.N)
        
        bank = set(s.c.YT.tolist())
        bank_matched = s.df.bank.tolist()
        assert set(bank_matched).issubset(bank)
        bank_unmatched = sorted(
            bank-set(bank_matched))

        rtk = set(s.c.YC.tolist())
        rtk_matched = s.df.rtk.tolist()
        assert set(rtk_matched).issubset(rtk)
        rtk_unmatched = sorted(rtk-set(rtk_matched))
        
        s.list = list(zip(bank_matched,rtk_matched))
        if N=='F' and s.c.a.use_unmatched:
            s.list += list(
                zip(bank_unmatched,rtk_unmatched))
            

    def __len__(s):
        return len(s.list)     
    
    
    def __getitem__(s, i): 
        M = i<len(s.df)
        bank = s.list[i][0]
        rtk = s.list[i][1]
        D = {'bank': bank, 'rtk': rtk}
        XT = s.c.XT[s.c.YT==bank]
        XC = s.c.XC[s.c.YC==rtk]
        YT = torch.ones(len(XT))*bank
        YC = torch.ones(len(XC))*bank ### 
        D['XT'] = XT
        D['YT'] = YT
        D['XC'] = XC
        D['YC'] = YC
        D['MT'] = torch.tensor([M]*len(XT))
        D['MC'] = torch.tensor([M]*len(XC))
        D['M'] = M
        return D
    
    