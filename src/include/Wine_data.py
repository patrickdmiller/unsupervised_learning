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
  _type = 'Wine'
  @classmethod
  def load_pickle(cls):
    return super().load_pickle(cls._type)
  
  def load(self, load_param):  
    # white['color']=0
    red= pd.read_csv(os.path.join(CONST.DATA_PATH, f'wine/winequality-red.csv'), dtype=np.float64, sep=';')
    red['color']=1
    self.df = red
    # self.df = pd.concat([white, red], axis=0, ignore_index=True)
    print('median', self.df['quality'].median(), 'mean', self.df['quality'].mean())
    self.df = self.df.assign(good=lambda x:x.quality//6)
# df.assign(AvgHalfBill=lambda x: x.AvgBill / 2)
    if self.verbose:
      print("loading", os.path.join(CONST.DATA_PATH, f'wine/.csv'))
    self.df = self.df.drop(['quality'], axis=1)
    #load col names
  def after_load(self):
    self.X_train = self.X_train.drop(['color'], axis=1)  
    self.X_test = self.X_test.drop(['color'], axis=1)  
    return super().after_load()
  
  def __init__(self,test_size = 0.3, verbose=False,scale=True, oversample=True, undersample=False, scale_type='ss'):
    super().__init__(
      test_size = test_size,
      verbose = verbose,
      scale=scale,
      label_col = 'good',
      oversample = oversample,
      name="Wine",
      scale_type=scale_type,
      undersample=undersample
    )


