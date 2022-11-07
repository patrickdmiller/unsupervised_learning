import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import pickle
from abc import ABC, abstractmethod
import os
from include import const as CONST
class Dataset(ABC):
  
  _type=None
  
  @classmethod
  def load_pickle(cls, file_name):
    print("loading pickle: ", os.path.join(CONST.DATA_PATH, 'pickles/', f'{file_name}.pickle'))
    file = open(os.path.join(CONST.DATA_PATH, 'pickles/', f'{file_name}.pickle'),'rb')
    object_file = pickle.load(file)
    print(object_file)
    file.close()  
    return object_file


  '''
  subclasses must implement load function that populates self.df or returns df
  '''
  @abstractmethod
  def load(self, load_param=None):
    pass
  
  def after_load(self):
    pass




  def __init__(self, test_size=0.3, verbose=False, oversample = False, undersample=False, scale = True, scale_type='ss', label_col = None, onehots = [], load_param=None, seed=42, name="Dataset"):
    self.verbose = verbose
    self.name = name
    self.load(load_param)
    self.models = {}
    if not label_col or label_col not in self.df:
      raise Exception("no label specified")
    
    if verbose:
      print(type(self))
    self.label_col = label_col
    self.do_onehot(col_names=onehots, drop=True)
    if scale:
      if scale_type == 'ss':
        self.scaler = StandardScaler()
      elif scale_type == 'mm':
        self.scaler = MinMaxScaler()
      if verbose:
        print("Scale type: ", self.scaler)
      self.do_scale()
    if verbose:
      print("all data positive : ", len(self.df[self.df[label_col] == 1]), "neg: ", len(self.df[self.df[label_col] == 0]))

    self.y = self.df[label_col].to_frame()
    self.X = self.df.drop([label_col], axis=1)
    
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=seed)
    
    #if custom logic after shuffling and splitting (like dropping an item)
    self.after_load()
    
    if verbose:
        print("split: Train: ", len(self.X_train), len(self.y_train), "Test: ", len(self.X_test), len(self.y_test))
        print("train positive : ", len(self.y_train[self.y_train[label_col] == 1]), "neg: ", len(self.y_train[self.y_train[label_col] == 0]))
    if oversample:
        _oversample =  RandomOverSampler(sampling_strategy='minority')
        self.X_train, self.y_train = _oversample.fit_resample(self.X_train, self.y_train)
    if undersample:
        _undersample = RandomUnderSampler(sampling_strategy='majority')
        self.X_train, self.y_train = _undersample.fit_resample(self.X_train, self.y_train)
    if verbose and (oversample or undersample):
        print("after resample\nsplit: Train: ", len(self.X_train), len(self.y_train), "Test: ", len(self.X_test), len(self.y_test))
        print("train positive : ", len(self.y_train[self.y_train[label_col] == 1]), "neg: ", len(self.y_train[self.y_train[label_col] == 0]))

    self.df_train = pd.concat([self.X_train, self.y_train], axis=1)

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
        # print(key, over_cols[key], type(over_cols[key]))
        if over_cols[key] > 1:
          cols.add(key)
      under_cols = self.df.min()
      for key in under_cols.keys():
        if under_cols[key] < -1:
          cols.add(key)
      to_scale = list(cols)
      if self.verbose:
        print("scaling", to_scale)
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
      self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=validation_percent_of_training, random_state=42) 
      print("after validation\n Train: ", len(self.X_train), len(self.y_train), "Val: ", len(self.X_val), len(self.y_val), "Test:", len(self.X_test), len(self.y_test))
      
      
  def pickle(self):
    with open(os.path.join(CONST.DATA_PATH, 'pickles/', f'{self.name}.pickle'), 'wb') as handle:
      pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)