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
                        default = None)

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

    parser.add_argument("--savemodel", 
                        "-s",
                        help="Save the model",
                        type=bool,
                        default=False)


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
dataloader_train, dataloader_test, dataloader_val = preprocess_classification_data(df)

# define the model
class runBNNClassification:
    def __init__(self, model, dataloader_train, dataloader_test, dataloader_val, device, epochs, lr, criterion, optimize, savemodel):
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
        self.savemodel = savemodel
        

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

            # get validation loss
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
            

            # get accuracy as a percentage
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in self.dataloader_val:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            accuracy_per = 100 * correct / total


            
            self.test_loss.append(test_loss)
            self.val_loss.append(val_loss)
            self.accuracy.append(100 * correct / total)

        print('Finished Training')

        return self.test_loss, self.val_loss, self.accuracy, accuracy_per

   
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


    def getEnsembleUncertainty(self):
        '''
        Function to get the uncertainty of the ensemble model
        '''
        # use the model to predict the validation data 3 times
        self.model.eval()
        ensemble_uncertainty = []
        with torch.no_grad():
            for X, y in self.dataloader_val:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                ensemble_uncertainty.append(outputs)
                outputs = self.model(X)
                ensemble_uncertainty.append(outputs)
                outputs = self.model(X)
                ensemble_uncertainty.append(outputs)

        # rescale the uncertainty to be between 0 and 1
        for i in range(len(ensemble_uncertainty)):
            ensemble_uncertainty[i] = torch.nn.functional.softmax(ensemble_uncertainty[i], dim=1)

        return ensemble_uncertainty



    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
def main():
    args = arg_inputs()

    if args.model == "SimpleFFBNNClassification":
        model = SimpleFFBNNClassification(4, 5)
        run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, args.epochs, args.lr, args.criterion, torch.optim.Adam, args.savemodel)
        test_loss, val_loss, accuracy, accuracy_per = run.train()
        print(f'Achieved Accuracy: {accuracy_per}')
        run.visualizeLoss()
        uncertainty_simple_model = run.get_uncertainty()
        #for i in range(len(uncertainty_simple_model)):
         #   for j in range(len(uncertainty_simple_model[i])):
          #      print(f'Max Uncertainty: {torch.max(uncertainty_simple_model[i][j])}')
           #     print(f'Min Uncertainty: {torch.min(uncertainty_simple_model[i][j])}')



        kl = run.model.kl_divergence()
        print(f'KL Divergence: {kl}')

       # get the ensemble uncertainty
        ensemble_uncertainty = run.getEnsembleUncertainty()
        
       # print the ensemble uncertainty for the first 5 samples
        for i in range(5):
            print(f'Ensemble Uncertainty: {ensemble_uncertainty[0][i]}')
            print(f'Ensemble Uncertainty: {ensemble_uncertainty[1][i]}')
            print(f'Ensemble Uncertainty: {ensemble_uncertainty[2][i]}')

            

        
        



        if args.savemodel:
            run.save_model('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/trainedModels/simple_model.pth')

        # run the simple model from terminal with the following command:
        # python runBNNClassification.py -m SimpleFFBNNClassification -e 1000 -l 0.0001 -c nn.CrossEntropyLoss() -s True


    else:
        model = LargeFFBNNClassification(4, 5)
        print(model)
        run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, args.epochs, args.lr, args.criterion, torch.optim.Adam, args.savemodel)
        test_loss, val_loss, accuracy, accuracy_per = run.train()
        print(f'Achieved Accuracy: {accuracy_per}')
        run.visualizeLoss()
        uncertainty_large_model = run.get_uncertainty()
        #for i in range(len(uncertainty_large_model)):
         #   for j in range(len(uncertainty_large_model[i])):
          #      print(f'Max Uncertainty: {torch.max(uncertainty_large_model[i][j])}')
           #     print(f'Min Uncertainty: {torch.min(uncertainty_large_model[i][j])}')

        kl = run.model.kl_divergence()
        print(f'KL Divergence: {kl}')

        if args.savemodel:
            run.save_model('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/trainedModels/large_model.pth')
        
    

if __name__ == "__main__":
    main()
