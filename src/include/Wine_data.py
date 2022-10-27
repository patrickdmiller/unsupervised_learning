'''
https://archive.ics.uci.edu/ml/datasets/wine+quality
'''

import pandas as pd
from include.Dataset import *
import numpy as np
from include import const as CONST
import math
import os
import re
class Wine(Dataset):
    
  def load(self, load_param):  
    white = pd.read_csv(os.path.join(CONST.DATA_PATH, f'wine/winequality-white.csv'), dtype=np.float64, sep=';')
    white['color']=0
    red= pd.read_csv(os.path.join(CONST.DATA_PATH, f'wine/winequality-red.csv'), dtype=np.float64, sep=';')
    red['color']=1
    self.df = pd.concat([white, red], axis=0)
    self.df = self.df.assign(good=lambda x:x.quality//7)
# df.assign(AvgHalfBill=lambda x: x.AvgBill / 2)
    if self.verbose:
      print("loading", os.path.join(CONST.DATA_PATH, f'wine/.csv'))
    self.df = self.df.drop(['quality'], axis=1)
    #load col names
  def after_load(self):
    self.X_train = self.X_train.drop(['color'], axis=1)  
    self.X_test = self.X_test.drop(['color'], axis=1)  
    return super().after_load()
  
  def __init__(self,test_size = 0.3, verbose=False,scale=True, oversample=True, load_param='combined'):
    super().__init__(
      test_size = test_size,
      verbose = verbose,
      scale=scale,
      label_col = 'good',
      oversample = oversample,
      load_param = load_param,
      name="Wine"
    )


