


from ..base import *



class IO:
    
    def save(s, x_name, file_name):
        print('save', file_name)
        name, ext = file_name.split('.')
        path = s.data_dir/file_name
        x = getattr(s, x_name)
        if isinstance(x, pd.DataFrame):
            x = x.reset_index()
            x = x.drop(columns='index')
            if ext=='csv':
                x.to_csv(path, index=False)         
            if ext=='feather':     
                x.to_feather(path)  
        elif ext=='pt':
            torch.save(x, path)
        delattr(s, x_name)
        gc.collect()   
        
        
    def load(s, file_name):
        print('load', file_name)
        name, ext = file_name.split('.')
        path = s.data_dir/file_name
        if ext=='csv':
            x = pd.read_csv(path)  
        if ext=='feather':
            x = pd.read_feather(path)
        if ext=='pt':
            x = torch.load(path)
        return x
    
    
    def remove(s, file_name):
        print('remove', file_name)
        os.remove(s.data_dir/file_name)        
               