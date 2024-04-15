import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class custom_data_loader(torch.utils.data.Dataset):

  def __init__(self, df,is_normalize = False):
    self.X = df.loc[:, df.columns != 'target']

    # make 'savings' numeric
    self.X['savings'] = np.where(self.X['savings'] == 'low', 0, np.where(self.X['savings'] == 'medium', 1, 2))

    if is_normalize:
        self.X = (self.X-self.X.mean())/self.X.std()
    
    #self.y = df.loc[:, df.columns == 'target']
    self.X = torch.FloatTensor(self.X.values.astype('float32'))
    
    # make y a tensor
    self.y = torch.FloatTensor(df.target.values)
    
    self.shape = self.X.shape

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
    return len(self.X)


class custom_data_loader_classification(torch.utils.data.Dataset):
  
    def __init__(self, df,is_normalize = False):
      self.X = df.loc[:, df.columns != 'target']
  
      # make 'savings' numeric
      self.X['savings'] = np.where(self.X['savings'] == 'low', 0, np.where(self.X['savings'] == 'medium', 1, 2))
  
      if is_normalize:
          self.X = (self.X-self.X.mean())/self.X.std()
      
      #self.y = df.loc[:, df.columns == 'target']
      self.X = torch.FloatTensor(self.X.values.astype('float32'))
      
      # make y a tensor
      self.y = torch.LongTensor(df.target.values)
      
      self.shape = self.X.shape
  
    def __getitem__(self, idx):
      return self.X[idx], self.y[idx]
  
    def __len__(self):
      return len(self.X)

# a function that preprocesses the data
def preprocess_data(df, batch_size):
    df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food.csv')
    df_custom = custom_data_loader(df_custom, is_normalize=True)
    scaler = StandardScaler()
    X = df_custom.X
    y = df_custom.y
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1,1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=11)
    dataset_train = TensorDataset(X_train, y_train)
    dataset_test = TensorDataset(X_test, y_test)
    dataset_val = TensorDataset(X_val, y_val)
    dataloader_train = DataLoader(dataset_train, batch_size= batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size= batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size= batch_size, shuffle=False)
    return dataloader_train, dataloader_test, dataloader_val

def preprocess_classification_data(df, batch_size):
  df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')
  df_custom = custom_data_loader_classification(df_custom, is_normalize=True)
  scaler = StandardScaler()
  X = df_custom.X
  y = df_custom.y
  #X = scaler.fit_transform(X)
  #y = scaler.fit_transform(y.reshape(-1,1))
 
  def make_zero_based(y):
    """Zero base the target variable"""
    for i in range(len(y)):
      y[i] = y[i] - 1
    return y
  y = make_zero_based(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)
  X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
  X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
  X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 11)

  dataset_train = TensorDataset(X_train, y_train)
  dataset_test = TensorDataset(X_test, y_test)
  dataset_val = TensorDataset(X_val, y_val)
  dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)
  dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle=False)
  dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle=False)
  return dataloader_train, dataloader_test, dataloader_val


def preprocess_classification_activeL_data(df):
    df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')
    df_custom = custom_data_loader_classification(df_custom, is_normalize=True)
    scaler = StandardScaler()
    X = df_custom.X
    y = df_custom.y

    def make_zero_based(y):
      """Zero base the target variable"""
      for i in range(len(y)):
        y[i] = y[i] - 1
      return y
    y = make_zero_based(y)

    #X = scaler.fit_transform(X)
    #y = scaler.fit_transform(y.reshape(-1,1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 11)
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
    dataset_train = TensorDataset(X_train, y_train)
    dataset_test = TensorDataset(X_test, y_test)
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)
    return dataloader_train, dataloader_test
