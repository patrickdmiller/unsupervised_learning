'''
https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++
'''

import pandas as pd
from include.Dataset import *
import numpy as np
from include import const as CONST
import os
import re
class Fire(Dataset):
    
  def load(self, load_param):   
    if self.verbose:
      print("loading", os.path.join(CONST.DATA_PATH, f'algerian_fire/data_{load_param}.csv'))
    self.df = pd.read_csv(os.path.join(CONST.DATA_PATH, f'algerian_fire/data_{load_param}.csv'), dtype=np.float64)
    print(self.df.columns)
    
    #load col names
      
  def __init__(self,test_size = 0.3, verbose=False,scale=True, oversample=True, load_param='combined'):
    super().__init__(
      test_size = test_size,
      verbose = verbose,
      scale=scale,
      label_col = 'Classes',
      oversample = oversample,
      load_param = load_param
    )


