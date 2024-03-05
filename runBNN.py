import torch
import GPUtil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from Utils import custom_data_loader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

from Models.simpleFFBNN import SimpleFFBNN


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
        self.loss = []
        #self.input_dim = len(data_test) # input_dim but have to think about this

    def train(self):
        for i in range(self.epoch):
            self.model.train()
            epoch_loss = 0
            for X, y in self.data_train:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(X)
            self.loss.append(epoch_loss / len(self.data_train.dataset))
            #print(f'Epoch {i + 1}, Loss: {loss.item()}')
            print(f'Epoch {i + 1}, Loss: {self.loss[-1]}')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for X, y in self.data_test:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                # for classification
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1
            print('Test Accuracy: ', round(correct / total, 3))

    def evaluate_regression(self, samples = 100, std_multiplier = 2):
        self.model.eval()
        X, y = next(iter(self.data_test))
        X, y = X.to(self.device), y.to(self.device)


        preds = [self.model(X) for i in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)
        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)
        ic_acc = (ci_lower <= y) * (ci_upper >= y)
        print(f'ic_acc after first transformation: {ic_acc}')
        ic_acc = ic_acc.float().mean()
        print(f'ic_acc after second transformation: {ic_acc}')
        return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

    #def predict(self, val_data):
    """ This function is for using the model to predict the validation data.
        will work on this later but leaving it here for logical structure purposes.
    """
     #   self.model.eval()
      #  with torch.no_grad():
       #     val_data = val_data.to(self.device)
        #    return self.model(val_data)

    #def save_model(self, path):
    """ This function is for saving the model to a file. 
             Not sure if I will use this but leaving it here for logical structure purposes. 
    """
     #   torch.save(self.model.state_dict(), path)
    
    def visualizeLoss(self):
        plt.plot(self.loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.show()




df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food.csv')

# print shape of x
#print(f' df_custom {df_custom.shape}')

df_custom = custom_data_loader(df_custom, is_normalize=True)
#print(f' df_custom {df_custom.shape}')

# scale the data
scaler = StandardScaler()

# fit transform the data
X = df_custom.X
print(f' X {X}')
y = df_custom.y

X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1,1))

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

# create dataloaders
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)

dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)


# run the simpleFFBNN with the X, y dataloader_train
#for epoch in range(2):
 #   for X, y in dataloader_train:
  #      print(f'Epoch: {epoch} X: {X} y: {y}')



# create the model
#regressor = SimpleFFBNN(input_dim = 4, output_dim =1)
# define optimizer need to clean later
#optimizer = optim.Adam(regressor.parameters(), lr=0.01)

# run instance of model class with the following arguments: self, model, data_train, data_test, epoch, lr, optimizer, criterion, device
run = runBNN(SimpleFFBNN(input_dim = 4, output_dim =1), dataloader_train, dataloader_test, 1000, 0.001, torch.optim.Adam, nn.MSELoss(), device)
# train model
run.train()
# test classification model
#run.test()
# visualize loss
run.visualizeLoss()

# evaluate regression
ic_acc, upper, lower = run.evaluate_regression()
print(f'IC Accuracy: {ic_acc.item()}, Upper: {upper.item()}, Lower: {lower.item()}')




""" def evaluate_regression(regressor,
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
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()


iteration = 0
for epoch in range(100):
    for i, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        loss = regressor.sample_elbo(inputs=datapoints, 
                                    labels=labels, 
                                    criterion= torch.nn.MSELoss(),
                                    sample_nbr=3,
                                    complexity_cost_weight=1/X_train.shape[0])

        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 10 == 0:
            ic_acc, upper, lower = evaluate_regression(regressor, X_test, y_test)
            print(f'Epoch: {epoch}, Iteration: {iteration}, Loss: {loss.item()}, IC Accuracy: {ic_acc.item()}, Upper: {upper.item()}, Lower: {lower.item()}') """