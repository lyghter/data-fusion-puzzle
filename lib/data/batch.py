


from ..base import *


@all_methods(staticmethod)
class Batch:
    def get_triplets(B):
        n = list(range(len(B['XT'])))
        perm = list(itertools.permutations(n,2))
        perm = torch.tensor(perm)
        pos = perm[:,0]
        neg = perm[:,1] 
        A = B['XT'][pos]
        P = B['XC'][pos]
        N = B['XC'][neg]
        return A,P,N
    
    
    def drop_unmatched(B):
        BM = {}
        BM['bank'] = B['bank'][B['M']]
        BM['XT'] = B['XT'][B['MT']]
        BM['YT'] = B['YT'][B['MT']]
        BM['rtk'] = B['rtk'][B['M']]        
        BM['XC'] = B['XC'][B['MC']]
        BM['YC'] = B['YC'][B['MC']]
        return BM

        
    def average(B, func):
        XT = []
        XC = []
        for uid in B['bank']:
            if func=='mean':
                XT.append(
                    B['XT'][B['YT']==uid].mean(0))
                XC.append(
                    B['XC'][B['YC']==uid].mean(0))
            if func=='median':
                XT.append(
                    B['XT'][B['YT']==uid].median(0).values)
                XC.append(
                    B['XC'][B['YC']==uid].median(0).values) 
        XT = torch.stack(XT)
        XC = torch.stack(XC)
        return XT,XC,B['bank']
        
        
    def concat(B):
        X = torch.cat([B['XT'],B['XC']])
        Y = torch.cat([B['YT'],B['YC']])
        return X,Y
        
        
    def collect(BB):
        R = dict(
            XT=[], XC=[],
            YT=[], YC=[],
            bank=[], rtk=[]
        )
        for B in BB:
            for k in R:
                R[k].append(B[k])
        for k in R:
            R[k] = torch.cat(R[k])
        return R
            
    
    
