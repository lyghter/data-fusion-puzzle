

from ..base import *

   
class DataSet2(Dataset):
    def __init__(s, d, a):
        s.d = d
        s.a = a
        bank = set(d.YT.tolist())
        bank_matched = d.P.bank.tolist()
        assert set(bank_matched).issubset(bank)
        bank_unmatched = sorted(
            bank-set(bank_matched))

        rtk = set(d.YC.tolist())
        rtk_matched = d.P.rtk.tolist()
        assert set(rtk_matched).issubset(rtk)
        rtk_unmatched = sorted(rtk-set(rtk_matched))
        
        s.list = list(zip(bank_matched,rtk_matched))
        if s.a.use_unmatched:
            s.list += list(
                zip(bank_unmatched,rtk_unmatched))
            

    def __len__(s):
        return len(s.list)     
    
    
    def __getitem__(s, i): 
        M = i<len(s.d.P)
        bank = s.list[i][0]
        rtk = s.list[i][1]
        D = {'bank': bank, 'rtk': rtk}
        XT = s.d.XT[s.d.YT==bank]
        XC = s.d.XC[s.d.YC==rtk]
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
    
    