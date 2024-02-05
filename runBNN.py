import torch
import GPUtil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

cuda = torch.cuda.is_available()
print("CUDA Available: ", cuda)

if cuda:
    gpu = GPUtil.getFirstAvailable()
    print("GPU Available: ", gpu)
    torch.cuda.set_device(gpu)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Device: ", device)

from simpleFFBNN import SimpleFFBNN


# a class that runs the BNN

class runBNN:
    def __init__(self, model, data_train, data_test, epoch, lr, optimizer, criterion, device):
        self.model = model
        self.data_train = data_train
        self.data_test = data_test
        self.epoch = epoch
        self.lr = lr
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()
        self.device = device
        self.model.to(device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)

    def train(self):
        for i in range(self.epoch):
            self.model.train()
            for X, y in self.data_train:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {i + 1}, Loss: {loss.item()}')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for X, y in self.data_test:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1
            print('Test Accuracy: ', round(correct / total, 3))



    
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


# print output of dataloaders
for X, y in dataloader_train:
    print(f'X: {X}')
    print(f'y: {y}')



# run the simpleFFBNN with the X, y dataloader_train
for epoch in range(2):
    for X, y in dataloader_train:
        print(f'Epoch: {epoch} X: {X} y: {y}')





# run the BNN
#model = SimpleFFBNN(dataloader_train, dataloader_train)
#print(model)

#run = runBNN(model, dataloader_train, dataloader_test, 100, 0.001, torch.optim.Adam, nn.MSELoss(), device)
#self, model, data_train, data_test, epoch, lr, optimizer, criterion, device