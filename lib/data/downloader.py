


from ..base import *
from .io import IO


@all_methods(verbose)
class Downloader(IO):
    def __init__(s, e):
        a = e.a
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
        s.tr = Reducer('DataFrame')(s.tr)
        c = 'timestamp'
        d = {'transaction_dttm':c}
        s.tr = s.tr.rename(columns=d)
        s.tr[c] = s.tr[c].progress_apply(
            pd.Timestamp)
        
        k = 2930
        s.tr_3k = s.tr[s.tr.user_id.isin(
            s.tr.user_id.unique()[:k])]
        assert s.tr_3k.user_id.nunique()==k
        s.save('tr_3k','tr.csv')
        s.save('tr','transactions.feather')
        shutil.copyfile(
            s.data_dir/'transactions.feather',
            s.data_dir/'_transactions.feather',     
        )
        
        
    def download_clickstreams(s): 
        path = '0554f0cf/clickstream.zip'
        s.cl = s.download_and_extract_zip(path)
        s.cl = Reducer('DataFrame')(s.cl)
        c = 'timestamp'
        s.cl[c]=s.cl[c].progress_apply(pd.Timestamp)
  
        k = 2463
        s.cl_3k = s.cl[s.cl.user_id.isin(
            s.cl.user_id.unique()[:k])]
        assert s.cl_3k.user_id.nunique()==k
        s.save('cl_3k','cl.csv')
        
        s.save('cl','clickstreams.feather')
        shutil.copyfile(
            s.data_dir/'clickstreams.feather',
            s.data_dir/'_clickstreams.feather',     
        )      
        
              
    def download_and_extract_zip(s,url):
        url = s.prefix + url
        name = url.split('/')[-1].split('.')[0]
        name_csv = f'{name}.csv'
        path_zip = s.data_dir/f'{name}.zip'
        path_feather = s.data_dir/f'{name}.feather' 
        download_large_file(url, path_zip)
        file = ZipFile(path_zip).open(name_csv)
        df = pd.read_csv(file)
        df = Reducer('DataFrame')(df)
        os.remove(path_zip)
        return df            
              
    
    def download_other_files(s):
        for i in tqdm(range(len(s.csv_urls))):
            url = s.prefix + s.csv_urls[i]
            path_csv = s.data_dir/url.split('/')[-1]
            df = pd.read_csv(url)
            df = Reducer('DataFrame')(df)
            df.to_csv(path_csv, index=False)
        
        
    def run(s):
        s.make_dirs()
        s.download_transactions()
        s.download_clickstreams()
        s.download_other_files()  
        
