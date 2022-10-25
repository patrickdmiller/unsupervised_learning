import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from abc import ABC, abstractmethod
import os

class Dataset(ABC):
  '''
  subclasses must implement load function that populates self.df or returns df
  '''
  @abstractmethod
  def load(self):
    pass

  def __init__(self, test_size=0.3, verbose=False, oversample = False, scale = True, scale_type=StandardScaler(), label_col = None, onehots = []):
    self.load()
    
    if not label_col or label_col not in self.df:
      raise Exception("no label specified")
    
    if verbose:
      print(type(self))
    self.label_col = label_col
    self.verbose = verbose
    self.do_onehot(col_names=onehots, drop=True)
    if scale:
      self.scaler = scale_type
      self.do_scale()
              
    self.y = self.df[label_col].to_frame()
    self.X = self.df.drop([label_col], axis=1)
    
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=1)
    
    if verbose:
        print("split: Train: ", len(self.X_train), len(self.y_train), "Test: ", len(self.X_test), len(self.y_test))
    if oversample:
        _oversample =  RandomOverSampler(sampling_strategy='minority')
        self.X_train, self.y_train = _oversample.fit_resample(self.X_train, self.y_train)
    if verbose:
        print("after resample\nsplit: Train: ", len(self.X_train), len(self.y_train), "Test: ", len(self.X_test), len(self.y_test))
    else:
        print("no resampling")
        
  def do_onehot(self, col_names, drop = True):
      #change strings to categorical floats. 
      for col_name in col_names:
        encoder = OneHotEncoder(sparse=True, categories='auto')
        transformer = encoder.fit_transform(self.df[[col_name]])
        names = encoder.categories_.copy()
        names = list(map(lambda x: col_name+'.'+x, names))
        self.df[names[0]] = transformer.toarray()
      if drop:
          self.df = self.df.drop(columns=col_names)
      
  def do_scale(self, _min = -10, _max = 10):
      if self.verbose:
        print("doing scaling")
        
      cols = set()
      over_cols = self.df.max()
      for key in over_cols.keys():
        if over_cols[key] > 10:
          cols.add(key)
          print("over > ", key, over_cols[key])
      under_cols = self.df.min()
      for key in under_cols.keys():
        if under_cols[key] < -10:
          print("under> ", key, under_cols[key])
          cols.add(key)
      to_scale = list(cols)
      self.df[to_scale] = self.scaler.fit_transform(self.df[to_scale])

  def describe(self):
    print("original data: ")
    print("\ttotal rows: ", len(self.X))
    # print("self.y", self.df)
    print("\ttotal true: ", len(self.df.loc[self.df[self.label_col] == 1]), "total false: ", len(self.df.loc[self.df[self.label_col] == 0]))

    print("resampled training data: ")
    print("\ttotal rows: ", len(self.X_train))
    # print("self.y", self.df)
    print("\tTRAIN:")
    print("\ttotal true: ", len(self.y_train.loc[self.y_train[self.label_col] == 1]), "total false: ", len(self.y_train.loc[self.y_train[self.label_col] == 0]))
    print("\tTEST:")
    print("\ttotal true: ", len(self.y_test.loc[self.y_test[self.label_col] == 1]), "total false: ", len(self.y_test.loc[self.y_test[self.label_col] == 0]))

  def generate_validation(self, validation_percent_of_training=0.2):
      self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=validation_percent_of_training, random_state=1) 
      print("after validation\n Train: ", len(self.X_train), len(self.y_train), "Val: ", len(self.X_val), len(self.y_val), "Test:", len(self.X_test), len(self.y_test))