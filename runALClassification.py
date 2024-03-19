from Utils import custom_data_loader_classification, preprocess_classification_data
import torch
import torch.nn as nn
import GPUtil
import pandas as pd
from Models.simpleFFBNNClassification import SimpleFFBNNClassification
from Models.largeFFBNNClassification import LargeFFBNNClassification
import matplotlib.pyplot as plt
import argparse

def arg_inputs():
    # initiate the parser
    parser = argparse.ArgumentParser()
    # add the arguments
    parser.add_argument("--model", 
                        "-m", 
                        help="The model to be used", 
                        type=str,
                        default="SimpleFFBNNClassification")

    parser.add_argument("--epochs",
                        "-e",
                        help="Number of epochs",
                        type=int,
                        default=1000)

    parser.add_argument("--lr",
                        "-l",
                        help="Learning rate",
                        type=float,
                        default=0.0001)
                    
    parser.add_argument("--criterion",
                        "-c",
                        help="Criterion",
                        type=str,
                        default="nn.CrossEntropyLoss()")


   # parse the arguments
    args = parser.parse_args()
    return args


def get_device():
    """Function to get the device to be used for training the model
    """
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
    return device

# get the device
device = get_device()

# read data and preprocess
df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')
print(df.head())
print(df.info())
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
        self.model.to(device)
        self.test_loss = []
        self.val_loss = []
        self.accuracy = []
        

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
            test_loss = 0
            with torch.no_grad():
                for X, y in self.dataloader_test:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    loss = self.criterion(outputs, y)
                    test_loss += loss.item()
            print(f'Epoch: {epoch}, Test Loss: {test_loss}')     
            print(f'Accuracy: {100 * correct / total}')

            self.model.eval()
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for X, y in self.dataloader_val:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    loss = self.criterion(outputs, y)
                    val_loss += loss.item()
            print(f'Epoch: {epoch}, Val Loss: {val_loss}')
            print(f'Accuracy: {100 * correct / total}')
            self.test_loss.append(test_loss)
            self.val_loss.append(val_loss)
            self.accuracy.append(100 * correct / total)

        print('Finished Training')

   
    def visualizeLoss(self):
        plt.plot(self.test_loss, label='Test Loss')
        plt.plot(self.val_loss, label='Val Loss')
        plt.legend()
        plt.show()

        plt.plot(self.accuracy, label='Accuracy')
        plt.legend()
        plt.show()


    def get_uncertainty(self):
        '''
        Function to get the uncertainty of the model predictions
        '''
        self.model.eval()
        uncertainty = []
        with torch.no_grad():
            for X, y in self.dataloader_val:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                uncertainty.append(outputs)

        # rescale the uncertainty to be between 0 and 1
        for i in range(len(uncertainty)):
            uncertainty[i] = torch.nn.functional.softmax(uncertainty[i], dim=1)


        return uncertainty

    
def main():
    args = arg_inputs()

    if args.model == "SimpleFFBNNClassification":
        model = SimpleFFBNNClassification(4, 5)
        run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, args.epochs, args.lr, args.criterion, torch.optim.Adam)
        run.train()
        run.visualizeLoss()
        uncertainty_simple_model = run.get_uncertainty()

    else:
        model = LargeFFBNNClassification(4, 5)
        run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, args.epochs, args.lr, args.criterion, torch.optim.Adam)
        run.train()
        run.visualizeLoss()
        
    

if __name__ == "__main__":
    main()
