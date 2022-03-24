


from ..base import *


def MRR(true, pred, k=100):
    
    assert 'bank' in true and 'rtk' in true
    assert 'bank' in pred and 'rtk_list' in pred
    
    df = true.copy(deep=True)
    df['rtk_list'] = pred['rtk_list'].apply(
        lambda l: l[:k])
              
    def get_MRR(r):
        if r['rtk'] in r['rtk_list']:
            mrr=1/(r['rtk_list'].index(r['rtk'])+1)
        else:
            mrr = 0
        return mrr            
        
    df['MRR'] = df.apply(get_MRR, axis=1)   
    MRR = df['MRR'].mean()  
    return MRR


def P(true, pred, k=100):
    
    assert 'bank' in true and 'rtk' in true
    assert 'bank' in pred and 'rtk_list' in pred
    
    df = true.copy(deep=True)
    df['rtk_list'] = pred['rtk_list'].apply(
        lambda l: l[:k])
    
    def get_P(r):
        if r['rtk'] in r['rtk_list']:
            p = 1
        else:
            p = 0
        return p                  
        
    df['P'] = df.apply(get_P, axis=1)
    P = df['P'].mean()
    return P


def predict_random_100(df):
    pred = pd.DataFrame()
    pred['bank'] = df['bank']

    def get_random_100(x):
        return df['rtk'].sample(100).tolist()

    pred['rtk_list'] = pred['bank'].apply(
        get_random_100)
    return pred
