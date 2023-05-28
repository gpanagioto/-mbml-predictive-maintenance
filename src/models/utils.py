import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd


def compute_error(trues: np.array, predicted: np.array, threshold: int):
    if threshold:
        predicted_thres = predicted[np.where(trues<threshold)]
        trues_thres  = trues[np.where(trues<threshold)[0]]
    else:
        print('No threshold')
        pass
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    try:
        return corr, mae, rae, rmse, r2, predicted_thres, trues_thres
    except:
         return corr, mae, rae, rmse, r2, predicted, trues

def get_data_for_component(data, component):
    cols = ['voltmean_3h', 'rotatemean_3h',
       'pressuremean_3h', 'vibrationmean_3h', 'voltsd_3h', 'rotatesd_3h',
       'pressuresd_3h', 'vibrationsd_3h', 'voltmean_24h', 'rotatemean_24h',
       'pressuremean_24h', 'vibrationmean_24h', 'voltsd_24h', 'rotatesd_24h',
       'pressuresd_24h', 'vibrationsd_24h', 'error1count', 'error2count',
       'error3count', 'error4count', 'error5count','age',
       'model_model1', 'model_model2', 'model_model3', 'model_model4','machineID', component]
    return data[cols]


def preprocess(X_init,test_size,classi=False,splitting=True):

        if splitting:
            X_train_unsc, X_test_unsc, y_train_unsc, y_test_unsc =split(X_init,test_size=test_size)
            X_init = X_init.to_numpy()
            y = X_init[:,-1]
            X = X_init[:,:-1]
        else:
            X_init = X_init.drop(columns=['machineID']).to_numpy()
            y = X_init[:,-1]
            X = X_init[:,:-1]
            X_train_unsc, X_test_unsc, y_train_unsc, y_test_unsc = train_test_split(X, y, test_size=test_size, random_state=0)
            
        print(X_train_unsc.shape,y_train_unsc.shape,X_test_unsc.shape,y_test_unsc.shape)
        X_mean = X_train_unsc.mean(axis=0)
        X_std = X_train_unsc.std(axis=0)

        y_std = y_train_unsc.std()
        y_mean = y_train_unsc.mean()

        X_train = (X_train_unsc - X_mean)/ X_std
        X_test = (X_test_unsc - X_mean)/X_std

        y_train = (y_train_unsc- y_mean)/ y_std 
        y_test = (y_test_unsc- y_mean)/y_std        

        X_train_torch = torch.tensor(X_train).float()
        y_train_torch = torch.tensor(y_train).float()
        if classi:
            y_train_torch = torch.tensor(y_train_unsc).float()
            y_train = y_train_unsc
            y_test = y_test_unsc
        X_test_torch = torch.tensor(X_test).float()

        return y, X, X_train_torch, y_train_torch,X_test_torch, X_test, y_test, X_train, y_train, y_std, y_mean

def split(data,test_size):
    
    machines = pd.read_csv('../data/raw/raw2/PdM_machines.csv')
    
    x_train_,x_test_,y_train_,y_test_,train_idx,test_idx = train_test_split(machines, machines.model, machines.machineID.index, test_size=test_size, stratify=machines.model, random_state=42)
    
    #print(y_test_.value_counts())
    #print(y_train_.value_counts())
    #print(machines.model.value_counts())
    training=data[data['machineID'].isin(train_idx)]
    training=training.drop(columns=['machineID']).to_numpy()
    test=data[data['machineID'].isin(test_idx)]
    test=test.drop(columns=['machineID']).to_numpy()

    
    x_train=training[:,:-1]
    y_train=training[:,-1]
    x_test=test[:,:-1]
    y_test=test[:,-1]

    return x_train,x_test,y_train,y_test