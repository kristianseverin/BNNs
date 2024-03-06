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
from Models.simpleFFBNN import SimpleFFBNN


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
        self.trainLoss = []
        self.testLoss = []
        self.valLoss = []
        #self.input_dim = len(data_test) # input_dim but have to think about this

    def train(self):
        for i in range(self.epoch):
            self.model.train()
            test_loss = 0
            train_loss = 0
            val_loss = 0
            for X, y in self.data_train:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                test_loss += loss.item() * len(X)

            self.model.eval()
            with torch.no_grad():
                for X, y in self.data_test:
                    X, y = X.to(self.device), y.to(self.device)
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    val_loss += loss.item()
            self.trainLoss.append(train_loss / len(self.data_train.dataset))
            self.testLoss.append(test_loss / len(self.data_train.dataset))
            self.valLoss.append(val_loss / len(self.data_train.dataset))
            #print(f'Epoch {i + 1}, Loss: {loss.item()}')
            print(f'Epoch {i + 1}, Train Loss: {self.trainLoss[-1]}, Test Loss: {self.testLoss[-1]}, Val Loss: {self.valLoss[-1]}')
            

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

    def evaluate_regression(self, regressor, X, y, samples = 100, std_multiplier = 2):
        self.model.eval()
        X, y = next(iter(self.data_test))
        X, y = X.to(self.device), y.to(self.device)
        preds = [self.model(X) for i in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        print(f'Means: {means}')
        stds = preds.std(axis=1)
        print(f'Stds: {stds}')
        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)
        ic_acc = (ci_lower <= y) * (ci_upper >= y)
        ic_acc = ic_acc.float().mean()
        return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()
        



    def predict(self, val_data):
        """ This function is for using the model to predict the validation data.
        """
        self.model.eval()
        with torch.no_grad():
            val_data = val_data.to(self.device)
            return self.model(val_data)

    
    def visualizeLoss(self):
        #plt.plot(self.loss, label='loss')
        plt.plot(self.trainLoss, label='train loss')
        #plt.plot(self.testLoss, label='test loss')
        plt.plot(self.valLoss, label='val loss')
        plt.legend()
        plt.show()

    def visualizePrediction(self, y, y_pred):
        # get them back to original scale
        y = scaler.inverse_transform(y)
        y_pred = scaler.inverse_transform(y_pred)

        plt.scatter(y, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])
        plt.show()

    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# read data
df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food.csv')

# custom data loader
df_custom = custom_data_loader(df_custom, is_normalize=True)

# scale the data
scaler = StandardScaler()

# fit transform the data
X = df_custom.X
y = df_custom.y

X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1,1))

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

# split into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=43)

# create dataloaders
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)
dataset_val = TensorDataset(X_val, y_val)

dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)
dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)

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
ic_acc, upper, lower = run.evaluate_regression(regressor = SimpleFFBNN(input_dim = 4, output_dim =1), X = X_val, y = y_val, samples = 100, std_multiplier = 2)
print(f'IC Accuracy: {ic_acc.item()}, Upper: {upper.item()}, Lower: {lower.item()}')


# predict
pred = run.predict(X_val)

# visualize prediction
run.visualizePrediction(y_val, pred)



# save model
run.save_model('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/trainedModels/FFBNN.pth')





#ic_acc, upper, lower = run.evaluate_regression()
#print(f'IC Accuracy: {ic_acc.item()}, Upper: {upper.item()}, Lower: {lower.item()}')

# predict
#pred = run.predict(X_val)

