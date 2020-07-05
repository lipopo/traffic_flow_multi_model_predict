import pandas as pd

from lib import Pipe


class XLSXReader(Pipe):
    def __init__(self, filename):
        self.data = pd.read_excel(filename)
    
    def concate(self, data):
        self.data = self.data.append(data)
