


from .base import *


class GitIgnore:
    def __init__(self, gitignore_path='.gitignore'):
        self.gitignore_path = gitignore_path
        
    def show(self, n_lines): 
        with open(self.gitignore_path, 'r') as f:
            s = f.read()
        l = s.split('\n')
        #n_lines = l.index('')
        pprint(l[:n_lines])
            
    def add(self, path:str):
        with open(self.gitignore_path, 'r+') as f:
            s = f.read()
            l = s.split('\n')
            if path not in l:
                f.seek(0, 0)
                f.write(path.rstrip('\r\n')+'\n'+s)
        
    def remove(self, path:str):
        with open(self.gitignore_path, 'r+') as f:
            s = f.read()
        l = s.split('\n')
        l = [p for p in l if p != path]
        s = '\n'.join(l)
        with open(self.gitignore_path, 'w+') as f:    
            f.write(s)
