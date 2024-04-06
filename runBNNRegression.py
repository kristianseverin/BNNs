import torch
import GPUtil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from Utils import custom_data_loader, preprocess_data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from Models.simpleFFBNN import SimpleFFBNN
from Models.denseBBBRegression import DenseBBBRegression
from Models.denseRegression import DenseRegressor
import argparse

def arg_inputs():
    # initiate the parser
    parser = argparse.ArgumentParser(description='Run BNN for regression')
    # add the arguments
    parser.add_argument("--model", 
                        "-m", 
                        help="The model to be used", 
                        type=str,
                        default="SimpleFFBNN")
    parser.add_argument("--dataloader_train",
                        "-dt",
                        help="Dataloader for train",
                        type=str,
                        default="dataloader_train")
    parser.add_argument("--dataloader_test",
                        "-dte",
                        help="Dataloader for test",
                        type=str,
                        default="dataloader_test")
    parser.add_argument("--dataloader_val",
                        "-dv",
                        help="Dataloader for val",
                        type=str,
                        default="dataloader_val")
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
                        default="nn.MSELoss()")
    parser.add_argument("--device",
                        "-d",
                        help="Device",
                        type=str,
                        default="device")

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
dataloader_train, dataloader_test, dataloader_val = preprocess_data(pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food.csv'))


# a class that runs the BNN
class runBNN:
    def __init__(self, model, data_train, data_test, data_val, epoch, lr, optimizer, criterion, device):
        self.model = model
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
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

    def objective(self, output, target, kl, beta):
        loss_fun = nn.MSELoss()
        discrimination_error = loss_fun(output.view(-1), target.view(-1))
        variational_bound = discrimination_error + beta * kl
        return variational_bound, discrimination_error, kl

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
                train_loss += loss.item() * len(X)

            self.model.eval()
            with torch.no_grad():
                for X, y in self.data_test:
                    X, y = X.to(self.device), y.to(self.device)
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    test_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                for X, y in self.data_val:
                    X, y = X.to(self.device), y.to(self.device)
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    val_loss += loss.item()


            
            self.trainLoss.append(train_loss / len(self.data_train.dataset))
            self.testLoss.append(test_loss / len(self.data_train.dataset))
            self.valLoss.append(val_loss / len(self.data_train.dataset))
            #print(f'Epoch {i + 1}, Loss: {loss.item()}')
            print(f'Epoch {i + 1}, Train Loss: {self.trainLoss[-1]}, Test Loss: {self.testLoss[-1]}, Val Loss: {self.valLoss[-1]}')


    def trainDenseBBB(self):
        for epoch in range(self.epoch):
            self.model.train()
            for X, y in self.data_train:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                Trainloss = self.model.sample_elbo(inputs=X,
                                               labels=y,
                                               criterion=self.criterion,
                                               sample_nbr=3,
                                               complexity_cost_weight=1/len(self.data_train))


            Trainloss.backward()
            self.optimizer.step()


            self.model.eval()
            with torch.no_grad():
                for X, y in self.data_test:
                    X, y = X.to(self.device), y.to(self.device)
                    Testloss = self.model.sample_elbo(inputs=X,
                                               labels=y,
                                               criterion=self.criterion,
                                               sample_nbr=3,
                                               complexity_cost_weight=1/len(self.data_test))
                    


            self.model.eval()
            with torch.no_grad():
                for X, y in self.data_val:
                    X, y = X.to(self.device), y.to(self.device)
                    Valloss = self.model.sample_elbo(inputs=X,
                                               labels=y,
                                               criterion=self.criterion,
                                               sample_nbr=3,
                                               complexity_cost_weight=1/len(self.data_val))

            self.trainLoss.append(Trainloss)
            self.testLoss.append(Testloss)
            self.valLoss.append(Valloss)

            print(f'Epoch {epoch + 1}, Train Loss: {self.trainLoss[-1]}, Test Loss: {self.testLoss[-1]}, Val Loss: {self.valLoss[-1]}')
            
    def trainClosedFormBNN(self):
        train_loss_closed, test_loss_closed, val_loss_closed = [], [], []
        train_log_likelihood_closed, test_log_likelihood_closed, val_log_likelihood_closed = [], [], []
        train_kl_closed, test_kl_closed, val_kl_closed = [], [], []
        m = len(self.data_train.dataset)  # number of samples
        train_R2 = []

        for epoch in range(self.epoch):

            self.model.train()

            outputs = self.model(self.data_train.dataset.tensors[0].to(self.device))
            loss, log_likelihood, scaled_kl = self.objective(outputs, self.data_train.dataset.tensors[1].to(self.device), self.model.kl_divergence(), 1 / m)
            train_loss_closed.append(loss)
            train_log_likelihood_closed.append(log_likelihood)
            train_kl_closed.append(scaled_kl)

            # get the R2 score
            y_pred = self.model(self.data_train.dataset.tensors[0].to(self.device))
            y_true = self.data_train.dataset.tensors[1].to(self.device)
            R2 = r2_score(y_true.detach().numpy(), y_pred.detach().numpy())
            train_R2.append(R2)

            loss.backward()
            self.optimizer.step()

            for layer in self.model.kl_layers:
                layer.clip_variances()

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.data_test.dataset.tensors[0].to(self.device))
                loss, log_likelihood, scaled_kl = self.objective(outputs, self.data_test.dataset.tensors[1].to(self.device), self.model.kl_divergence(), 1 / m)
                test_loss_closed.append(loss.item())
                test_log_likelihood_closed.append(log_likelihood.item())
                test_kl_closed.append(scaled_kl.item())

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.data_val.dataset.tensors[0].to(self.device))
                loss, log_likelihood, scaled_kl = self.objective(outputs, self.data_val.dataset.tensors[1].to(self.device), self.model.kl_divergence(), 1 / m)
                val_loss_closed.append(loss.item())
                val_log_likelihood_closed.append(log_likelihood.item())
                val_kl_closed.append(scaled_kl.item())

            

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss_closed[-1]}, Test Loss: {test_loss_closed[-1]}, Val Loss: {val_loss_closed[-1]}, Train R2: {train_R2[-1]}')


            self.trainLoss = train_loss_closed
            self.testLoss = test_loss_closed
            self.valLoss = val_loss_closed
            print(f'val_loss: {val_loss_closed}')

        


    # for regression
    def evaluate_regression(self, regressor, X, y, data_test, samples = 100, std_multiplier = 3):
        self.model.eval()
        X, y = next(iter(self.data_test))
        X, y = X.to(self.device), y.to(self.device)
        preds = [self.model(X) for i in range(samples)]
        preds = torch.stack(preds)
        RMSE = torch.sqrt(((preds.mean(axis=0) - y) ** 2).mean())
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)
        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)
        ic_acc = (ci_lower <= y) * (ci_upper >= y)
        ic_acc = ic_acc.float().mean()
        return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean(), ci_upper, ci_lower, RMSE
        


    def predict(self, val_data):
        """ This function is for using the model to predict the validation data.
        """
        self.model.eval()
        with torch.no_grad():
            for X, y in val_data:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                return output
    
    def visualizeLoss(self):
        #plt.plot(self.loss, label='loss')
        plt.plot(self.testLoss, label='train loss')
        #plt.plot(self.testLoss, label='test loss')
        plt.plot(self.valLoss, label='val loss')
        plt.legend()
        plt.show()

    def visualizePrediction(self, y, y_pred):
        plt.scatter(y, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.title('True vs Predicted values') 
        #plt.xlim([0, plt.xlim()[1]])
        #plt.ylim([0, plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])
        plt.show()

    def visualizeMetrics(self, ci_lower, ci_upper, y, y_pred):
        # make a density plot
        ci_upper = ci_upper.detach().numpy()
        ci_lower = ci_lower.detach().numpy()
        sns.distplot(y, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'True Values')
        sns.distplot(y_pred, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'Predicted Values')
        #sns.distplot(ci_upper, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'CI Upper')
        #sns.distplot(ci_lower, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'CI Lower')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.title('True vs Predicted values')
        plt.legend()
        plt.show()

    def saveMetrics(self, path, pred, y_val, RMSE):

        RMSE = RMSE.detach().numpy()


        train_loss = torch.tensor(self.trainLoss).detach().numpy()
        
        # save metrics to numpy
        np.save(f'{path}/preds.npy', pred)
        np.save(f'{path}/true.npy', y_val)
        np.save(f'{path}/RMSE.npy', RMSE)        

        np.save(f'{path}/train_loss.npy', train_loss)
        np.save(f'{path}/test_loss.npy', self.testLoss)
        np.save(f'{path}/val_loss.npy', self.valLoss)

    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

def main():
    args = arg_inputs()

    # get the device
    args.device = get_device()

    # get the optimizer
    args.optimizer = optim.SGD

    # get the criterion
    args.criterion = nn.MSELoss()
    
    if args.model == 'SimpleFFBNN':
        run = runBNN(SimpleFFBNN(input_dim = 4, output_dim = 1), dataloader_train, dataloader_test, dataloader_val, args.epochs, args.lr, args.optimizer, args.criterion, args.device)
        run.trainClosedFormBNN()
        run.visualizeLoss()
        ic_acc, upper, lower, ci_upper, ci_lower, RMSE = run.evaluate_regression(regressor = SimpleFFBNN(input_dim = 4, output_dim =1), X = next(iter(dataloader_test))[0], y = next(iter(dataloader_test))[1], data_test = dataloader_test, samples = 100, std_multiplier = 2)
        print(f'IC Accuracy: {ic_acc.item()}, Upper: {upper.item()}, Lower: {lower.item()}, RMSE = {RMSE}')
    
        # get the predictions
        pred = run.predict(dataloader_val)
        
        # visualize the predictions

        y_val = next(iter(dataloader_val))[1]

        run.visualizePrediction(y_val, pred)

        run.visualizeMetrics(ci_lower, ci_upper, y_val, pred)

        #run.saveMetrics('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/SimpleFFBNN', pred, y_val, RMSE)

        # get kl divergence
        kl = run.model.kl_divergence()
        print(f'KL Divergence: {kl}')

    elif args.model == 'DenseBBBRegression':
        run = runBNN(DenseBBBRegression(input_dim = 4, output_dim = 1), dataloader_train, dataloader_test, dataloader_val, args.epochs, args.lr, args.optimizer, args.criterion, args.device)
        run.trainDenseBBB()
        run.visualizeLoss()
        ic_acc, upper, lower, ci_upper, ci_lower, RMSE = run.evaluate_regression(regressor = DenseBBBRegression(input_dim = 4, output_dim =1), X = next(iter(dataloader_test))[0], y = next(iter(dataloader_test))[1], data_test = dataloader_test, samples = 100, std_multiplier = 2)
        print(f'IC Accuracy: {ic_acc.item()}, Upper: {upper.item()}, Lower: {lower.item()}, RMSE: {RMSE}')

        # get the predictions
        pred = run.predict(dataloader_val)

        # visualize the predictions
        y_val = next(iter(dataloader_val))[1]
        run.visualizePrediction(y_val, pred)

        run.visualizeMetrics(ci_lower, ci_upper, y_val, pred)

        run.saveMetrics('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/Regression/BBBDenseRegression', pred, y_val, RMSE)

        # get the KL divergence
        kl = run.model.kl_divergence()
        print(f'KL Divergence: {kl}')

    else:
        run = runBNN(DenseRegressor(input_dim = 4, output_dim = 1), dataloader_train, dataloader_test, dataloader_val, args.epochs, args.lr, args.optimizer, args.criterion, args.device)
        run.train()
        run.visualizeLoss()
        ic_acc, upper, lower, ci_upper, ci_lower, RMSE = run.evaluate_regression(regressor = DenseRegressor(input_dim = 4, output_dim =1), X = next(iter(dataloader_test))[0], y = next(iter(dataloader_test))[1], data_test = dataloader_test, samples = 100, std_multiplier = 2)
        print(f'IC Accuracy: {ic_acc.item()}, Upper: {upper.item()}, Lower: {lower.item()}, RMSE: {RMSE}')

        # get the predictions
        pred = run.predict(dataloader_val)

        # visualize the predictions
        y_val = next(iter(dataloader_val))[1]
        run.visualizePrediction(y_val, pred)

        run.visualizeMetrics(ci_lower, ci_upper, y_val, pred)

        run.saveMetrics('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/DenseRegression', pred, y_val, RMSE)

        # get the KL divergence
        kl = run.model.kl_divergence()
        print(f'KL Divergence: {kl}')



if __name__ == '__main__':
    main()

# run this in the terminal with the following command:
# python runBNRegression.py -m SimpleFFBNN -dt dataloader_train -dte dataloader_test -dv dataloader_val -e 1000 -l 0.0001 -c nn.MSELoss() -d device
# run dense model from terminal with the following command:
# python runBNNRegression.py -m DenseBBBRegression -dt dataloader_train -dte dataloader_test -dv dataloader_val -e 1000 -l 0.0001 -c nn.MSELoss() -d device