import torch
import pandas as pd
import numpy as np

class custom_data_loader(torch.utils.data.Dataset):

  def __init__(self, df,is_normalize=False):
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
