'''
https://archive-beta.ics.uci.edu/ml/datasets/spambase#Attributes
'''

import pandas as pd
from include.Dataset import *

from include import const as CONST
import os
import re
class Spam(Dataset):
    
  def load(self, load_param=None):
    with open(os.path.join(CONST.DATA_PATH, 'spam/spambase.names')) as f:
      lines = f.readlines()
    lines = list(map(lambda l: re.sub(':.+\n?','', l), lines))
    lines+=['is_spam']
    self.df = pd.read_csv(os.path.join(CONST.DATA_PATH, 'spam/spambase.data'))
    self.df.columns = lines
    self.df.head
    #load col names
      
  def __init__(self,test_size = 0.3, verbose=False,scale=True, oversample=True):

    super().__init__(
      test_size = test_size,
      verbose = verbose,
      scale=scale,
      label_col = 'is_spam',
      oversample = oversample
    )


