



from ..base import *
from ..data.batch import Batch
from . import metric

from pytorch_metric_learning import losses, miners, distances


def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float()
    sum_embeddings = torch.sum(
        last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class Model(pl.LightningModule):
    def __init__(s, a, c):
        super().__init__()
        s.c = c
        s.a = a
        vocab_size = len(s.c.event_encoder)
        
        max_len = max(a.bank_len, a.rtk_len)
        n = s.a.bb_pp['block_size']
        mpe = (max_len//n+(max_len%n>0))*n
        config = tfs.BigBirdConfig(
            vocab_size = vocab_size,
            max_position_embeddings = mpe,
            pad_token_id = 0,
            bos_token_id = -1,#2,
            eos_token_id = -1,#1,
            sep_token_id = -1,
            **a.bb_pp
        )
        s.enc = tfs.BigBirdModel(config)
            
        Loss = getattr(losses, a.loss)
        s.loss = Loss(**a.loss_pp)
            
        if a.miner:
            s.miner = getattr(miners, a.miner)(
            **a.miner_pp
        )
        euclidean = [
            'MarginLoss',
            'TripletMarginLoss',
        ]
        if a.loss in euclidean:
            s.distance = distances.LpDistance(
                normalize_embeddings = True, 
                p = 2, 
                power = 1,
            )
            s.largest = False
        cosine = [
            'CircleLoss'
        ]
        if a.loss in cosine:
            s.distance=distances.CosineSimilarity()
            s.largest = True


            
    def forward(s, B):
        for k in ['XT','XC']:
            X = B[k]
            M = (B[k]!=0).long()             
            X = s.enc(
                input_ids = X,
                attention_mask = M,
            )
            X = X.last_hidden_state 
            X = mean_pooling(X,M)
            B[k] = X
        return B

    
    def get_loss(s, B):
        X,Y = Batch.concat(B)
        if s.a.miner:
            T = s.miner(X,Y)
            loss = s.loss(X,Y,T)
        else:
            loss = s.loss(X,Y)
        return loss
                            
        
    def training_step(s, B, batch_idx):
        B = s(B)
        loss = s.get_loss(B)
        return loss

                    
    def validation_step(s, B, batch_idx):
        bs = s.a.val_batch_size
        B = s(B)
        loss = s.get_loss(B)
        s.log('loss', loss, batch_size=bs)
        B['XT'],B['XC'],_ = Batch.average(
            B, s.a.avg_pred)
        return B

    
    def predict_step(s, B, batch_idx):
        B = s(B)
        B['XT'],B['XC'],_ = Batch.average(
            B, s.a.avg_pred)
        return B
    
    
    def get_rtk_lists(s, R, DF, k=None):
        assert 'rtk' in DF
        k = min(s.a.k, len(R['XC']))
        
        IDS = s.distance(R['XT'],R['XC'])\
            .topk(k, largest=s.largest)\
            .indices.tolist()
        
        rtk_lists = []      
        for ids in IDS:
            l = DF['rtk'].iloc[ids].tolist()
            rtk_lists.append(l)
        return rtk_lists
       
        
    def validation_epoch_end(s, BB):
        R = Batch.collect(BB)
        val = s.c.train.V.iloc[:len(R['XT'])]
        true = val 
        pred = pd.DataFrame()
        pred['bank'] = val['bank']
        
        k = s.a.k
        pred['rtk_list'] = s.get_rtk_lists(
            R, val, k
        )
#         pred = metric.predict_random_100(true)
        P = metric.P(true, pred, k)
        MRR = metric.MRR(true, pred, k)
        R1 = 2*MRR*P/(MRR+P)

        s.log('R1', R1)
        s.log('MRR', MRR)
        s.log('P', P)        
            
            
    def predict_epoch_end(s, BB):
        R = Batch.collect(BB)
        test = s.c.test.P
        pred = pd.DataFrame()
        pred['bank'] = test['bank']
        pred['rtk_list'] = s.get_rtk_lists(
            R, test, k=100)
        func = s.c.uid_encoder.inverse_transform
        c = 'rtk_list'
        pred[c] = pred[c].apply(func)
#         if s.a.docker:
#             pred[c] = pred[c].apply(
#                 lambda l: ['0']+l[:-1])
        c = 'bank'
        pred[c] = func(pred[c])
        return pred

    
    def on_validation_model_train(s):
        p = s.a.log_dir
        p /= s.a.exp_name
        p /= 'version_0'
        p /= 'metrics.csv'
        cc = ['loss','R1','MRR','P']
        df = load(p)[cc].round(4)
        print(df)
    

    def configure_optimizers(s):
        return torch.optim.AdamW(
            s.parameters(), lr=s.a.lr)
    
    

