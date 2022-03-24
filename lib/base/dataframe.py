



from .base import *


def get_int_dtypes():
    dtypes = []
    for n in [64,32,16,8]:
        for p in ['','u']:
            dtypes.append(
                getattr(np,f'{p}int{n}'))
    return dtypes


def is_ok(series, dtype):
    iinfo = np.iinfo(dtype)
    first = iinfo.min < series.min() < iinfo.max
    second = iinfo.min < series.max() < iinfo.max
    return first and second


def reduce_memory_usage(df):
    dtypes = get_int_dtypes()
    int_cc = df.select_dtypes(
        include=dtypes).columns
    for c in int_cc:
        best_dtype = np.int64
        for dtype in dtypes:
            if is_ok(df[c], dtype):
                best_dtype = dtype
        df[c] = df[c].astype(best_dtype)
    float_cc = df.select_dtypes(
        include=[np.float64,np.float32]).columns
    for c in float_cc:
        df[c] = df[c].astype(np.float16)
        
          