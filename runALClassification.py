import torch
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
    df, df_active = train_test_split(df, test_size=0.9, random_state=42)
    df.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df.csv', index = False)
    df_active.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_active.csv', index = False)
    return df

def trainSeedModel():
    
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
    
    print(f'Uncertainty: {uncertainty}')

    


    


    
def main():
    args = arg_inputs()
    device = get_device()
    activeLearning()


if __name__ == '__main__':
    main()

# run from terminal
# python runALClassification.py --model simple --savemodel True