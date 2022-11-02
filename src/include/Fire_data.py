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
  _type = 'Fire'
  @classmethod
  def load_pickle(cls):
    return super().load_pickle(cls._type)
  
  def load(self, load_param):   
    if self.verbose:
      print("loading", os.path.join(CONST.DATA_PATH, f'algerian_fire/data_{load_param}.csv'))
    self.df = pd.read_csv(os.path.join(CONST.DATA_PATH, f'algerian_fire/data_{load_param}.csv'), dtype=np.float64)

    # self.df = self.df.drop(['month', 'day','year'], axis=1)
    # self.df.insert(0, '_ID', range(0, 0 + len(self.df)))
    #load col names
      
  def __init__(self,test_size = 0.3, verbose=False,scale=True, oversample=True, undersample=False, load_param='combined', seed=42):
    super().__init__(
      test_size = test_size,
      verbose = verbose,
      scale=scale,
      label_col = 'Classes',
      oversample = oversample,
      undersample = undersample,
      load_param = load_param,
      name="Fire",
      seed=seed
    )


