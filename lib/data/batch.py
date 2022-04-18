


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

        
    def average_TC(B, func):
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
        B['XT'] = torch.stack(XT)
        B['XC'] = torch.stack(XC)
        return B

        
    def average(B, func):
        XT = []
        XC = []
        B['bank'] = []
        B['rtk'] = []
        for uid,label in zip(B['uid'],B['label']):
            
            if label=='bank':
                x,y = 'XT','YT'
            if label=='rtk':
                x,y = 'XC','YC' 
                
            B[label].append(uid)  
                
            f = getattr(B[x][B[y]==uid], func)
            
            if func=='mean':
                X = f(0)
            if func=='median':
                X = f(0).values
                
            if label=='bank':
                XT.append(X)
            if label=='rtk':
                XC.append(X)                
                
        if XT!=[]:
            XT = torch.stack(XT)
        if XC!=[]:
            XC = torch.stack(XC)
        B['XT'] = XT
        B['XC'] = XC
        return B
    
    
    def concat(B):
        X = torch.cat([B['XT'],B['XC']])
        Y = torch.cat([B['YT'],B['YC']])
        return X,Y
        
        
    def collect_TC(BB):
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
    
    
    def collect(BB):
        kk = [
            'XT','XC','YT','YC',
            'uid','label','bank','rtk',
        ]
        R = {k:[] for k in kk}
        special_keys=['uid','label']+['bank','rtk']
        for B in BB:
            for k in R:
                if k in special_keys:
                    R[k] += B[k]
                else:
                    R[k].append(B[k])
        for k in R:
            if k in special_keys:
                R[k] = np.array(R[k])
            else:
                R[k] = torch.cat(R[k])
#         R['XT'] = R['X'][R['label']=='bank']
#         R['XC'] = R['X'][R['label']=='rtk']
        return R            
    
    
