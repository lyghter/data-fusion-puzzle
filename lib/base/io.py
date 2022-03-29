

from .base import *


def load(path): 
    path = Path(path)
    if path.suffix == '.json':
        with open(path,'r') as f: x = json.load(f)
    if path.suffix == '.csv':
        x = pd.read_csv(path)
    if path.suffix == '.feather':
        x = pd.read_feather(path)
    if path.suffix == '.pth': 
        x = torch.load(
            path, map_location = 'cuda' \
                if torch.cuda.is_available() else 'cpu')
    return x
            
            
def save(x,path): 
    path = Path(path)
    if path.suffix == '.json':
        with open(path,'w') as f: json.dump(x,f)  
    if path.suffix == '.csv':
        x.to_csv(path)
    if path.suffix == '.feather':
        x = x.reset_index()
        x = x.drop(columns='index')
        x.to_feather(path)
    if path.suffix == '.pth': 
        x = torch.save(x, path)
    
    