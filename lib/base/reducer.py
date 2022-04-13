



from .base import *


class Reducer:
    def __init__(s, x_type_name):
        if x_type_name=='DataFrame':
            import numpy as np
            s.lib = np
        if x_type_name=='Tensor':
            import torch 
            s.lib = torch        
        s.x_type_name = x_type_name
        
        
    def __call__(s, x):
        if s.x_type_name=='DataFrame':
            assert isinstance(x, pd.DataFrame)
            s.transform_dataframe(x)         
        if s.x_type_name=='Tensor':
            assert isinstance(x, torch.Tensor)
            s.transform_tensor(x)     
        
        
    def transform_dataframe(s, df):
        dtypes = s.get_int_dtypes()
        int_cc = df.select_dtypes(
            include=dtypes).columns
        for c in int_cc:
            best_dtype = np.int64
            for dtype in dtypes:
                if s.is_ok(df[c], dtype):
                    best_dtype = dtype
            df[c] = df[c].astype(best_dtype)
        float_cc = df.select_dtypes(
            include=[np.float64,np.float32]).columns
        for c in float_cc:
            df[c] = df[c].astype(np.float16)
        
        
    def transform_tensor(s, tensor):
        if type(tensor.flatten()[0].item()): 
            dtypes = s.get_int_dtypes()
            best_dtype = torch.int64
            for dtype in dtypes:
                if s.is_ok(df[c], dtype):
                    best_dtype = dtype
            tensor = tensor.to(best_dtype)
        
        
    def get_int_dtypes(s):
        dtypes = []
        for n in [64,32,16,8]:
            for p in ['','u']:
                try:
                    dtypes.append(
                        getattr(s.lib,f'{p}int{n}'))
                except:
                    pass
        return dtypes    
        
        
    def is_ok(s, x, dtype):
        iinfo = s.lib.iinfo(dtype)
        first = iinfo.min < x.min() < iinfo.max
        second = iinfo.min < x.max() < iinfo.max
        return first and second        
            


