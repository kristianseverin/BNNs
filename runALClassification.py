import torch
from torch.distributions import Categorical
import pandas as pd
import numpy as np
import GPUtil
import argparse
import matplotlib.pyplot as plt
from Utils import custom_data_loader_classification, preprocess_classification_data
from Utils import preprocess_classification_data
from Models.largeFFBNNClassification import LargeFFBNNClassification
from Models.simpleFFBNNClassification import SimpleFFBNNClassification
from runBNNClassification import get_device, arg_inputs, custom_data_loader_classification, runBNNClassification
from sklearn.model_selection import train_test_split


def splitActiveData(df):
    '''This function splits the data into a training set and an active set. 
       The training set is used to train the seed model and the active set is used in the active learning process.
       The function saves the datasets to csv files. The active df does not contain the target values.
       
       Args:
       df (pd.DataFrame): The dataframe with the data.
       '''
    df, df_active = train_test_split(df, test_size=0.9, random_state=42)
    df.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df.csv', index = False)
    # remove the target values from the active data
    df_active = df_active.drop(columns = ['target'])
    df_active.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_active.csv', index = False)
    return df

def trainSeedModel():
    '''This function trains the seed model on the data that is not used in the active learning process. 
       The seed model is used to predict the target values of the samples with the highest uncertainty.'''
    
    # read data
    df = splitActiveData(pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv'))

    # preprocess data
    dataloader_train, dataloader_test, dataloader_val = preprocess_classification_data(df)

    # get the device
    device = get_device()

    # define the model
    model = SimpleFFBNNClassification(4, 5)
    run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, 350, 0.0001, torch.nn.CrossEntropyLoss(), torch.optim.SGD, False)

    # train the model
    run.train()

    # visualize the loss
    run.visualizeLoss()

    # save the model
    torch.save(model, '/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model.pth')

    # load the model
    model = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model.pth')
    
    return model

def activeLearning():
    '''This function implements the active learning process. The function uses the seed model to predict the target values of the samples with the highest uncertainty.
       The function then simulates the target values for the samples with the highest uncertainty using the same logic that was used, when the data was generated in the first place AKA the oracle.
       The function then adds the samples with the highest uncertainty to the training set and retrains the model. The process is repeated until the model has been trained on all the data.

       TOdo: use train the model for x rounds and then evaluate the model on the val set. If the model performs better when the samples with the highest uncertainty are added to the training set, then save the best model (measured by best accuracy).
        repeat the process for desired number of rounds. Save the model with the highest accuracy on the val set. 
       '''

    # if model exists, load it, else train it
    try:
        model = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model.pth')
    except:
        model = trainSeedModel()
    
    # read data
    df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_active.csv')

    # get the device
    device = get_device()

    # preprocess data
    dataloader_train, dataloader_test, dataloader_val = preprocess_classification_data(df)

    # load the seed model
    model = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model.pth')

    # use the model to predict the target values
    model.eval()
    predictions = []
    uncertainty = []
    with torch.no_grad():
        for X, y in dataloader_train:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted)
            uncertainty.append(outputs)

    # get the uncertainty
    for i in range(len(uncertainty)):
        uncertainty[i] = torch.nn.functional.softmax(uncertainty[i], dim=1)

    
    def acquisitionFunction(uncertainty):
        '''This function calculates the entropy and variation ratio of the model predictions.
        More uncertainty means higher entropy and variation ratio. More acquisition functions can be added.

        Args:
        uncertainty (list): A list of tensors with the model predictions as probabilities.
        '''
        uncertainty = torch.cat(uncertainty, dim=0)
        print(f'Uncertainty: {uncertainty}')
        print(f'Uncertainty shape: {uncertainty.shape}')
        entropy = Categorical(probs=uncertainty).entropy()
        print(f'Entropy: {entropy}')
        print(f'Entropy shape: {entropy.shape}')

        #entropy = uncertainty.entropy()
        variation_ratio = 1 - torch.max(uncertainty, dim=1).values
        return entropy, variation_ratio
        

    entropy, variation_ratio = acquisitionFunction(uncertainty)

    # get the indices of the samples with the highest uncertainty
    n = 100
    indices_entropy = np.argsort(entropy)[:n]
    indices_variation_ratio = np.argsort(variation_ratio)[:n]

    # get the samples with the highest uncertainty
    df_entropy = df.iloc[indices_entropy]
    df_variation_ratio = df.iloc[indices_variation_ratio]

    def simulateOracle(df):
        '''This function simulates the target values for the samples with the highest uncertainty using the same logic that was used, when the data was generated in the first place.
           The function is used to simulate the oracle/human expert in the active learning process.

        Args:
        df (pd.DataFrame): The dataframe with the samples with the highest uncertainty.

        '''

        # get the features
        income = df['monthly_income']
        time = df['time_of_month']
        savings = df['savings']
        guests = df['guests']

        quality_based_on_income = np.where(
            income >= 7000, 5, 
            np.where(income >= 4000, 4, 
            np.where(income >= 3000, 3, 
            np.where(income >= 2000, 2, 1)))
            )


        # Determine quality based on time of month
        quality_based_on_time = np.where(time >= 16, 1, 5)


        # Determine quality based on size of savings
        quality_based_on_savings = np.where(savings == 'high', 5, np.where(savings == 'medium', 3, 1))

        # Determine quality based on number of guests
        quality_based_on_guests = np.where(
            guests == 0, 3, 5
        )

        # Calculate the overall quality of the food exactly like the data was generated in the first place but no noise
        quality_of_food = (quality_based_on_income * 0.4 + quality_based_on_time * 0.1 + quality_based_on_savings * 0.2 + quality_based_on_guests * 0.3) / 1

        # add the target to the dataframe
        df['target'] = quality_of_food.astype(int)

        # make the target 0-indexed
        df['target'] = df['target'] - 1
        
        return df

    df_entropy = simulateOracle(df_entropy)
    print(f'df_entropy: {df_entropy}')
    print(f'unique values: {df_entropy["target"].unique()}')    

    # add the samples with the highest uncertainty to the training set
    df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df.csv')
    df = pd.concat([df, df_entropy], ignore_index=True)
    

def main():
    args = arg_inputs()
    device = get_device()
    activeLearning()


if __name__ == '__main__':
    main()

# run from terminal
# python runALClassification.py --model simple --savemodel True