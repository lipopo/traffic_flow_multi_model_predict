import pandas as pd

from lib import Pipe


class XLSXReader(Pipe):
    def __init__(self, filename, *args, **kwargs):
        self.data = pd.read_excel(filename, *args, **kwargs)
    
    def convert_index_to_datetime(self):
        self.data.index = pd.to_datetime(self.data.index)
        return self
    
    def concate(self, data):
        self.data = self.data.append(data)
    
    def save(self, fname):
        self.data.to_excel(fname)
