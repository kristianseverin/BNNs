from Utils import custom_data_loader_classification, preprocess_classification_data
import torch
import torch.nn as nn
import GPUtil
import pandas as pd
from Models.simpleFFBNNClassification import SimpleFFBNNClassification


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

# read data and preprocess
df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')
dataloader_train, dataloader_test, dataloader_val = preprocess_classification_data(df)

# define the model
class runBNNClassification:
    def __init__(self, model, dataloader_train, dataloader_test, dataloader_val, device, epochs, lr, criterion, optimizer):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.dataloader_val = dataloader_val
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for X, y in self.dataloader_train:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch: {epoch}, Loss: {running_loss}')

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in self.dataloader_test:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            print(f'Accuracy: {100 * correct / total}')
        print('Finished Training')

                

        

# run the model
model = SimpleFFBNNClassification(4, 5)
run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, 1000, 0.0001, nn.CrossEntropyLoss(), torch.optim.Adam)

run.train()


    