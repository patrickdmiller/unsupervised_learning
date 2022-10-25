
import pandas as pd
from include.Dataset import *

from include import const as CONST
import os

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

class Framingham(Dataset):
    
    def load(self):
        self.df = pd.read_csv(os.path.join(CONST.DATA_PATH, 'framingham/data.csv'))
        
    def __init__(self,test_size = 0.3, verbose=False,scale=True, oversample=True):

        super().__init__(
            test_size = test_size,
            verbose = verbose,
            scale=scale,
            label_col = 'TenYearCHD',
            oversample = oversample
        )


