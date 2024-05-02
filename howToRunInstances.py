# import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import repeat

import numpy as np
import pandas as pd
from Utils import custom_data_loader, preprocess_data, preprocess_activeL_data
from Utils.SummaryWriter import LogSummary
from Models.simpleFFBNN import SimpleFFBNN
from Models.denseRegression import DenseRegressor
from Models.paperModel import SimpleFFBNNPaper

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import SubsetRandomSampler

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import entropy

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

device = get_device()

model = SimpleFFBNNPaper(4, 1)

class SaveOutput():
    def __init__(self, instances, batch_size, rounds):
        self.T = instances
        self.batch_size = batch_size
        self.outputs = []
        self.rounds = rounds
        self.counter = 0


    def __call__(self, module, module_in, module_out):
        if self.counter < 3:
            sample_data = np.random.randint(self.batch_size)
            #outs = module_out.view(self.batch_size, -1)
            outs = module_out.view(self.T, self.batch_size, -1)[:, 0, :]
            layer_size = outs.shape[1]

            
            write_summary.per_round_layer_output(layer_size, outs, self.rounds)
            
            # print the output of the layer
            
            self.counter += 1


    def clear(self):
        self.outputs = []
        

dataset_train, dataset_test, dataset_activeL, df_custom = preprocess_activeL_data(pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food.csv'))

class runActiveLearning():
    def __init__(self, model_name, model, top_unc, dataloader_train, dataloader_test, dataset_active_l, epochs, rounds, learning_rate, 
    batch_size, instances, seed_sample, retrain, resume_round, optimizer, df_custom):
        self.model_name = model_name
        self.model = model
        self.top_unc = top_unc
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.dataset_active_l = dataset_active_l
        self.epochs = epochs
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.instances = instances
        self.seed_sample = seed_sample
        self.optimizer = optimizer
        self.df_custom = df_custom
        

        # a set of lists to store the selected indices with highest uncertainty
        self.selected_data = set([])
        # unexplored data
        self.unexplored_data = set(range(len(dataloader_train)))

        # make sure sklearn.metrics.r2_score is imported
        #self.r2_score = r2_score


    
    def objective(self, output, target, kl, beta):
        '''Objective function to calculate the loss function / KL divergence'''
        loss_fun = nn.MSELoss()
        discrimination_error = loss_fun(output.view(-1), target)
        variational_bound = discrimination_error + beta * kl
        return variational_bound, discrimination_error, kl

    def get_entropy(self, y):
        '''Function to calculate the entropy of the ensemble outputs
        y: the ensemble outputs (shape: 30, 64, 1)'''
        # calculate the entropy of the ensemble outputs using pytorch
        flattened_y = y.view(y.size(0), -1)

        probs = F.softmax(flattened_y, dim=1)


        entropy = -(probs * torch.log(probs)).sum(dim=1)

        #entropy = entropy.detach().numpy()

        #H = entropy(y, axis=1) # calculate the entropy of the ensemble outputs using scipy.stats.entropy
        # return H as a Tensor
        return entropy

    def get_validation_data(self, is_validation):
        if not is_validation:
            # train sampler randomly samples data from the selected data set
            train_sampler = SubsetRandomSampler(list(self.selected_data))
            # train loader will load the data from the train sampler
            self.train_loader = DataLoader(self.dataloader_train, batch_size=self.batch_size, sampler=train_sampler, num_workers=1)

        indices = list(self.unexplored_data)
        np.random.shuffle(indices)
        split = int(np.floor(0.1 * len(indices)))  # this line is to split the training_data into 90% training and 10% validation
        validation_idx = np.random.choice(indices, size = split) # this line is to randomly select 10% of the data for validation
        train_sampler = SubsetRandomSampler(list(self.selected_data))
        validation_sampler = SubsetRandomSampler(validation_idx)
        self.train_loader = DataLoader(self.dataloader_train, batch_size=self.batch_size, sampler=train_sampler, num_workers=1)
        self.validation_loader = DataLoader(self.dataloader_train, batch_size=self.batch_size, sampler=validation_sampler, num_workers=1)

    def random_data(self, rounds):
        if rounds == 0:    
            # randomly select data
            self.selected_data = set(range(self.dataloader_train))  # seed sample in Rakeesh & Jain paper
           
            #self.unexplored_data = self.unexplored_data.difference(self.selected_data) # all 

        else:
            minimum_index = np.random.choice(list(self.unexplored_data), self.top_unc)
      
            self.selected_data = self.selected_data.union(minimum_index)

            self.unexplored_data = self.unexplored_data.difference(self.selected_data)



    def activeDataSelection(self, rounds):

        
        if rounds == 1:

            self.selected_data = set(range(len(self.dataloader_train)))
            self.unexplored_data = self.selected_data
            print(f'Length of the unexplored data: {len(self.unexplored_data)}, round: {rounds}')

        else:
            self.all_data = DataLoader(self.dataloader_train, batch_size=self.batch_size, shuffle=False, num_workers=1)
            print(f'Length of the all data: {len(self.all_data)} in round: {rounds}')
            correct = 0
            metrics = []
            hook_handles = []
            save_output = SaveOutput(self.instances, self.batch_size, self.rounds)
            self.model.eval()
            for layer in self.model.kl_layers:
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)

            with torch.no_grad():
                for batch_index, (X, y) in enumerate(self.all_data):
                    batch_size = X.shape[0]
                    save_output.batch_size = batch_size
                    X = X.repeat(self.instances, 1)
                    y = y.squeeze()
                    y = y.repeat(self.instances)
                    X, y = X.to(device), y.to(device)
                    y_pred = self.model(X)
       
                    ensemble_outputs = y_pred.reshape(self.instances, batch_size, 1)
                    entropy = self.get_entropy(ensemble_outputs)
                    metrics.append(entropy)
                    
                save_output.clear()
                save_output.counter = 0
                for handle in hook_handles:
                    handle.remove()

                metrics = torch.cat(metrics)
                new_indices = torch.argsort(metrics, descending=True).tolist()
                new_indices = [n for n in new_indices if n not in self.selected_data]
            
                self.selected_data =  set(new_indices[:self.top_unc])
                self.unexplored_data = self.unexplored_data.difference(self.selected_data)
                print(f'Length of the unexplored data: {len(self.unexplored_data)}, round: {rounds}')
                print(f'Length of the selected data: {len(self.selected_data)}, round: {rounds}')
        
        #return self.selected_data
                




     
             
  
    def annotateSelectedData(self, rounds):
        
        print(f'Length of the selected data: {len(self.selected_data)}')
        indices = list(self.selected_data)
        print(f'in the annotateSelectedData function, the indices are: {indices}')

        data_to_annotate = [self.all_data.dataset[i] for i in indices]

        # remove the selected data from the all data
        x_all = [x for x, y in self.all_data.dataset]
        y_all = [y for x, y in self.all_data.dataset]
        
        x_all = [x for i, x in enumerate(x_all) if i not in indices]
        y_all = [y for i, y in enumerate(y_all) if i not in indices]
        print(f'Length of the x_all: {len(x_all)}')
        print(f'Length of the y_all: {len(y_all)}')

        # create a new dataset from the remaining data
        self.dataloader_train = TensorDataset(torch.stack(x_all), torch.stack(y_all))
        print(f'Length of the new dataset: {len(self.dataloader_train)}')


        
        def refit_and_rescale(data):
            
            data_to_fit_X = self.df_custom.X
            data_to_fit_y = self.df_custom.y

            scaler = StandardScaler().fit(data_to_fit_X)
            # get the x_values from the data to be annotated
            
            x_values = [x for x, y in data]
            y_values = [y for x, y in data]

            x_arrays =[x.numpy() for x in x_values]
            y_arrays = [y.numpy() for y in y_values]
            
            x_descaled = [torch.tensor(scaler.inverse_transform(x.reshape(1, -1))) for x in x_arrays]

            # get the x_values in numpy format
            x_np = [x.numpy() for x in x_descaled]
            x_flattened = [arr.flatten() for arr in x_np]

            # create a dataframe from the x_values
            x_df = pd.DataFrame(x_flattened, columns = ['income', 'time', 'savings', 'guests'])

            '''due to the way the data is transformed and inverse transformed 0 value are not exactly 0, but very close to 0.
            Therefore, the values close to 0 are replaced with 0'''
            tolerance = 1e-5
            x_df = x_df.mask(x_df.abs() < tolerance, 0)

            return x_df

        
        df = refit_and_rescale(data_to_annotate)


        def determine_quality(data):

            '''quality of food is determined by the income, time, savings and guests 
            following the approach the data was originally generated with.
            This function is therefore acting as the oracle.
            args:
            data: the data to be annotated in a pandas dataframe format with the columns income, time, savings and guests
            
            nb: the y_values should be generated without random noise as noise doesn't make theoretical sense in this context'''
            quality_based_on_income = np.where(data['income'] >= 7000, 5,
            np.where(data['income'] >= 4000, 4,
            np.where(data['income'] >= 3000, 3,
            np.where(data['income'] >= 2000, 2, 1))))
            quality_based_on_time = np.where(data['time'] >= 16, 1, 5)
            quality_based_on_savings = np.where(data['savings'] == 2, 5,
            np.where(data['savings'] == 1, 3, 1))
            quality_based_on_guests = np.where(data['guests'] == 0, 3, 5) 
            quality_of_food = (quality_based_on_income * 0.4 + quality_based_on_time * 0.1 + quality_based_on_savings * 0.2 + quality_based_on_guests * 0.3) / 1

            # make the quality of food y in the data
            data['quality_of_food'] = quality_of_food

            return data

            
        annotated_data = determine_quality(df)

        
        # scale the newly annotated data
        x_scaler = StandardScaler().fit(self.df_custom.X)
        x = annotated_data.drop('quality_of_food', axis = 1)
        y = annotated_data['quality_of_food']
        x_scaled = x_scaler.transform(x)
        x_scaled = torch.tensor(x_scaled.astype(np.float32))   
        y_scaler = StandardScaler().fit(self.df_custom.y.reshape(-1, 1))
        y_scaled = y_scaler.transform(y.values.reshape(-1, 1))
        y_scaled = torch.tensor(y_scaled)

        # create a tensor dataset from the annotated data

        '''Note to myself tomorrow:
        access the X,y of both datasets and make one dataset from them'''

        
        x_already_annotated = [x for x, y in self.dataset_active_l]
        y_already_annotated = [y for x, y in self.dataset_active_l]
        
        x_alr_numpy = [x.numpy() for x in x_already_annotated]
        x_alr_tensors = torch.tensor(x_alr_numpy)

        y_alr_numpy = [y.numpy() for y in y_already_annotated]
        y_alr_tensors = torch.tensor(y_alr_numpy)

    
        x_all_annotated = torch.cat((x_scaled, x_alr_tensors), 0)
        y_all_annotated = torch.cat((y_scaled, y_alr_tensors), 0)


        combined_dataset = TensorDataset(x_all_annotated, y_all_annotated)    
        print(f'combined dataset: {combined_dataset}, {len(combined_dataset)}')
        
        # should be fed into the train() function as the new training data every round
        self.all_annotated_data = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle = False, num_workers=1)
        print(f'Length of the all annotated data: {len(self.all_annotated_data)} should be longer than the previous length of the all annotated data')

    def TrainModel(self, rounds, epochs, is_validation):
        '''This function trains the seed model for the active learning process
        '''
        
        '''have to change train_loader to active_data_l'''
        
        #print('running model')
        t_total, v_total = 0, 0
        t_r2_scores = []
        if epochs == 1:
            self.get_validation_data(is_validation)
        self.model.train()
        t_loss, v_loss = [], []
        t_likelihood, v_likelihood = [], []
        t_kl, v_kl = [], []
        self.model.train()
        m = len(self.train_loader)
       # print(f'this is the train loader: {self.train_loader}, {len(self.train_loader)}')

       # print('before loop, this is the train loader: {}'.format(self.train_loader), len(self.train_loader))
        for batch_index, (inputs, targets) in enumerate(self.train_loader):
         #   print('running loop')
            X = inputs.repeat(self.instances,1) # (number of mcmc samples, input size)
            #Y = targets.squeeze()
            Y = Y.repeat(self.instances) # (number of mcmc samples, output size)
            X, Y = X.to(device), Y.to(device)
            outputs = self.model(X)
            loss, log_likelihood, kl = self.objective(outputs, Y, self.model.kl_divergence(), 1 / m)
            t_likelihood.append(log_likelihood.item())
            t_kl.append(kl.item())
            t_total += targets.size(0)

            

            #outputs = outputs.view(self.batch_size, -1)
          
            # calculate r2 score manually
            r2_score_value = 1 - (np.sum((outputs.detach().cpu().numpy() - targets.detach().cpu().numpy()) ** 2) / np.sum((targets.detach().cpu().numpy() - np.mean(targets.detach().cpu().numpy())) ** 2))
            t_r2_scores.append(r2_score_value)
            
            t_loss.append(loss.item())
            loss.backward()

            # define the optimizer
            optimizer = self.optimizer

            optimizer.step()
            for layer in self.model.kl_layers:
                layer.clip_variances()
        
        if is_validation:
            #print(f'this is the validation data {self.validation_loader}, these are the characteristics {len(self.validation_loader)}')
            m_val = len(self.validation_loader)
            self.model.eval()
            for batch_index, (inputs, targets) in enumerate(self.validation_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss_val, log_likelihood_val, kl_val = self.objective(outputs, targets, self.model.kl_divergence(), 1 / m_val)
                v_total += targets.size(0)
                v_loss.append(loss_val.item())
                v_likelihood.append(log_likelihood_val.item())
                v_kl.append(kl_val.item())

            
            avg_v_loss = np.average(v_loss)
            avg_t_loss = np.average(t_loss)
            avg_v_likelihood = np.average(v_likelihood)
            avg_t_likelihood = np.average(t_likelihood)
            avg_v_kl = np.average(v_kl)
            avg_t_kl = np.average(t_kl)


            print(
                'epochs: {}, train loss: {}, train likelihood: {}, train kl: {}'.format(
                    epochs, avg_t_loss, \
                    avg_t_likelihood, avg_t_kl))

            print(
                'epochs: {}, validation loss: {}, validation likelihood: {}, validation kl: {}'.format(
                    epochs, avg_v_loss, \
                    avg_v_likelihood, avg_v_kl))

            return avg_v_loss

        else:
            avg_t_loss = np.average(t_loss)
            avg_t_likelihood = np.average(t_likelihood)
            avg_t_kl = np.average(t_kl)
            avg_t_r2 = np.average(t_r2_scores)

         #   print(
          #      'epochs: {}, train loss: {}, train likelihood: {}, train kl: {}, train_avg_R2: {}'.format(
           #         epochs, avg_t_loss, \
            #        avg_t_likelihood, avg_t_kl, avg_t_r2))

            return avg_t_loss, avg_t_r2

    
    def TestModel(self, rounds):
        if device.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)

        self.model.load_state_dict(state['weights'])
        print(f'Model loaded: {self.model}')

        self.model.eval()
        predictions = []
        actual = []
        mse_scores = []
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(self.dataloader_test):
                X, Y = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)

                # Calculate the MSE loss for the batch
                mse_loss = nn.MSELoss()
                loss = mse_loss(outputs, Y)

                # Get the MSE score as a Python scalar
                mse_score = loss.item()
                mse_scores.append(mse_score)

                # Convert predictions and actual values to numpy arrays
                predictions.append(outputs.detach().cpu().numpy())      
                actual.append(Y.detach().cpu().numpy())
                

        predictions = np.concatenate(predictions)
        actual = np.concatenate(actual)
        df = pd.DataFrame(data = {'Predictions': predictions, 'Actual': actual})
        df.loc['R2'] = 1 - np.sum((df.Actual - df.Predictions) ** 2) / np.sum((df.Actual - np.mean(df.Actual)) ** 2)
        df.loc['MSE'] = mean_squared_error(df.Actual, df.Predictions)
        
        #print('Non-Ensemble Test MSE:{:.3f}, TestR2:{:.3f}'.format(df.loc["MSE"][0], df.loc["R2"][0]))
                

    def getTrainedModel(self, rounds):
        # path to save the trained model
        self.train_weight_path = 'trainedModels/trained_weights/' + self.model_name + '_' + 'e' + str(self.epochs) + '_' + '-r' + str(rounds) + '-b' + str(self.batch_size) + '.pkl'
        return (self.model, self.train_weight_path)


    def saveModel(self, model, optimizer, path_to_save):
        state = {
            'rounds': self.rounds,
            'weights': model.state_dict(),
            'selected_data': self.selected_data,
            'optimizer': self.optimizer.state_dict()
            }

        path_to_save = 'trainedModels/trained_weights/' + self.model_name + '_' + 'e' + str(self.epochs) + '_' + '-r' + str(self.rounds) + '-b' + str(self.batch_size) + '.pkl'

        torch.save(state, path_to_save)
        
if __name__ == '__main__':
    if not os.path.isdir('trainedModels/trained_weights'):
        os.makedirs('trainedModels/trained_weights')


    # use the class to run the active learning
    active_learning = runActiveLearning(model_name='simple', model=model, dataloader_train=dataset_train, top_unc = 10, dataloader_test=dataset_test, dataset_active_l= dataset_activeL, epochs=100, rounds=7, learning_rate=0.001, batch_size=64, instances = 30, seed_sample=4, retrain=False, resume_round=False, optimizer= torch.optim.Adam(model.parameters(), lr=0.001), df_custom = df_custom)

    write_summary = LogSummary('active_learning')

    # get data to train the model
    active_learning.get_validation_data(is_validation=True)

    # train just the seed model
    active_learning.TrainModel(1, 5, True)

    # get the trained model
    model, path = active_learning.getTrainedModel(1)

    # save the model
    active_learning.saveModel(model, active_learning.optimizer, path)

    # run the active learning process
    for r in range(1, active_learning.rounds):
        print(f'Round: {r}')
            #model, path = active_learning.getTrainedModel(r)
                 
        print(f'Training model in round: {r}')
        active_learning.activeDataSelection(r) 
        print(f'Annotating selected data in round: {r}')

        if r == 1 or r == 2:
            pass

        else:
            active_learning.annotateSelectedData(r)

            # save a model for the round
        #    active_learning.saveModel(model, active_learning.optimizer, path)

        #    model, path = active_learning.getTrainedModel(r)

        # train the model
        #print(f'Training model in round: {r}')
        #active_learning.TrainModel(r, 5, False)
