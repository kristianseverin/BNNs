import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from simpleFFBNN import SimpleFFBNN

# load data
df = pd.read_csv('quality_of_food.csv')

# convert 'savings' column to numeric
df['savings'] = np.where(df['savings'] == 'low', 0, np.where(df['savings'] == 'medium', 1, 2))

# predictors and target
X = df.drop(columns=['quality_of_food'])
y = df[['quality_of_food']]

# scale the data
scaler = StandardScaler()

X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

# create dataloaders
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)

dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

print(f' dataloadertrain {dataloader_train}')
print(f' dataloadertest {dataloader_test}')

print(X_train.shape[1])
print(y_train.shape[1])

def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 100,
                        std_multiplier = 2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()

    # rmse
    rmse = torch.sqrt(torch.mean((means - y)**2))

    # mae
    mae = torch.mean(torch.abs(means - y))

    # mse
    mse = torch.mean((means - y)**2)

    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean(), rmse, mae, mse



regressor = SimpleFFBNN(4, 1)

# define optimizer
#optimizer = optim.Adam(regressor.parameters(), lr=0.01)  # only works with cuda
#criterion = torch.nn.MSELoss()

# train the model
