


from ..base import *
from .io import IO


@all_methods(verbose)
class Downloader(IO):
    def __init__(s, a):
        s.data_dir = a.data_dir
        s.prefix = 'https://storage.yandexcloud.net/datasouls-ods/materials/'
        s.csv_urls = [
            'acfacf11/train_matching.csv',
            'b949c04c/mcc_codes.csv',
            '705abbab/click_categories.csv',
            'e33f2201/currency_rk.csv',

            'b99fed70/puzzle.csv',
            'f76e8087/sample_submission.csv',
        ]
        s.zip_urls = [
            '0433a4ca/transactions.zip',
            '0554f0cf/clickstream.zip',
        ]
        s.clickstream_cc = [
            'user_id',
            'cat_id',
            'new_uid',
            'timestamp', 
        ]
        s.transactions_cc = [
            'user_id',
            'mcc_code',
            'currency_rk',
            'transaction_amt',
            'transaction_dttm',
        ]
    
    
    def make_dirs(s):
        shutil.rmtree(
            s.data_dir, ignore_errors=bool(1))
        s.data_dir.mkdir()
    
            
    def download_transactions(s):
        s.tr = s.download_and_extract_zip(
            '0433a4ca/transactions.zip')
        reduce_memory_usage(s.tr)
        c = 'timestamp'
        d = {'transaction_dttm':c}
        s.tr = s.tr.rename(columns=d)
        s.tr[c] = s.tr[c].progress_apply(
            pd.Timestamp)
        s.save('tr','transactions.feather')
        
        
    def download_clickstreams(s): 
        path = '0554f0cf/clickstream.zip'
        gcl = s.download_and_extract_zip(path)
        reduce_memory_usage(gcl)
        gcl = gcl.groupby('user_id')
        uids = list(gcl.groups.keys())
        n_groups = len(uids)
        
        k = 0        
        dfs = []
        uids = []
        for i,(uid,df) in enumerate(tqdm(gcl)):
            i += 1
            df = df.drop(columns='new_uid')
            c = 'timestamp'
            df[c] = df[c].apply(pd.Timestamp)
            dfs.append(df)
            uids.append(uid)
            k += 1
            if k==5000 or i==n_groups:
                s.df = pd.concat(dfs)
                s.save('df', f'{i}.feather') 
                io.save(
                    uids, s.data_dir/f'{i}.json')
                k = 0
                dfs = []
                uids = []
            
                
    def download_and_extract_zip(s,url):
        url = s.prefix + url
        name = url.split('/')[-1].split('.')[0]
        name_csv = f'{name}.csv'
        path_zip = s.data_dir/f'{name}.zip'
        path_feather = s.data_dir/f'{name}.feather' 
        download_large_file(url, path_zip)
        file = ZipFile(path_zip).open(name_csv)
        df = pd.read_csv(file)
        reduce_memory_usage(df)
        os.remove(path_zip)
        return df            
              
    
    def download_other_files(s):
        for i in tqdm(range(len(s.csv_urls))):
            url = s.prefix + s.csv_urls[i]
            path_csv = s.data_dir/url.split('/')[-1]
            df = pd.read_csv(url)
            reduce_memory_usage(df)
            df.to_csv(path_csv, index=False)
        
        
    def run(s):
        s.make_dirs()
        s.download_transactions()
        s.download_clickstreams()
        s.download_other_files()  
        
