import numpy as np
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
import pandas as pd
from typing import Tuple
from src.models.models import poisson_model, linear_model, heteroscedastic_model 
from src.models.utils import compute_error
import csv
import os
from __init__ import root_dir
class SVI_model_format():
    '''
    defining the type of the regression
    '''
    poisson = {'name':"POISSON", 'model':poisson_model}
    heterosc = {'name':"HETEROSCEDASTIC", 'model':heteroscedastic_model}
    linear = {'name':"LINEAR", 'model':linear_model}

class SVI_regression_model():
    '''
    SVI regression: choosing the correct data, preprocess them, pyro inference and prediction
    '''
    def __init__(self, data: pd.DataFrame, component: str) -> None:
        
        self.data = data
        self.component = component

    def get_data_for_component(self) -> pd.DataFrame:
        
        '''
        returns the feautures of the dataset and the component we want to predict for
        '''
    
        cols = ['voltmean_3h', 'rotatemean_3h',
                'pressuremean_3h', 'vibrationmean_3h', 'voltsd_3h', 'rotatesd_3h',
                'pressuresd_3h', 'vibrationsd_3h', 'voltmean_24h', 'rotatemean_24h',
                'pressuremean_24h', 'vibrationmean_24h', 'voltsd_24h', 'rotatesd_24h',
                'pressuresd_24h', 'vibrationsd_24h', 'error1count', 'error2count',
                'error3count', 'error4count', 'error5count','age',
                'model_model1', 'model_model2', 'model_model3', 'model_model4', self.component]

        return self.data[cols]
    
    def preprocess(self, X_init: pd.DataFrame, model: str) -> \
        Tuple[np.ndarray, np.ndarray,  torch.tensor,  torch.tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.float64, np.float64]:

        '''
        the necessary data preprocess before procceding in pyro inference
        '''
        print(f"{model} Regression")

        X = X_init.to_numpy()

        # Keep the last column as target y
        y = X[:,-1]
        X = X[:,:-1]

        # standardize input features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X = (X - X_mean) / X_std

        # standardize pickups
        y_mean = y.mean()
        y_std = y.std()
        y = (y - y_mean) / y_std

        train_perc = 0.80 # percentage of training data
        split_point = int(train_perc*len(y))
        perm = np.random.permutation(len(y))
        ix_train = perm[:split_point]
        ix_test = perm[split_point:]
        X_train = X[ix_train,:]
        X_test = X[ix_test,:]
        y_train = y[ix_train]
        y_test = y[ix_test]

        # Prepare data for Pyro model
        X_train_torch = torch.tensor(X_train).float()

        if model == SVI_model_format.poisson['name']:
            y_train_torch = torch.tensor(y_train * y_std + y_mean).int()
        else:
            y_train_torch = torch.tensor(y_train).float()
           
        return y, X, X_train_torch, y_train_torch, X_test, y_test, X_train, y_train, y_std, y_mean
    
    def pyro_inference(self, X_train_torch: torch.tensor, y_train_torch: torch.tensor, model: object, steps: int):
        '''
        pyro inference
        '''

        # Define guide function
        guide = AutoDiagonalNormal(model)

        # Reset parameter values
        pyro.clear_param_store()

        # Define the number of optimization steps
        n_steps = steps

        # Setup the optimizer
        adam_params = {"lr": 0.0001} # learning rate (lr) of optimizer
        optimizer = ClippedAdam(adam_params)

        # Setup the inference algorithm
        elbo = Trace_ELBO(num_particles=1)
        svi = SVI(model, guide, optimizer, loss=elbo)

        # Do gradient steps
        for step in range(n_steps):
            elbo = svi.step(X_train_torch, y_train_torch)
            if step % 100 == 0:
                print("[%d] ELBO: %.1f" % (step, elbo))

        return guide
    
    def post_process(self, guide: object, model: object, X_train_torch: torch.tensor, y_train_torch: torch.tensor, X_test: np.ndarray, y_test: np.ndarray, y_std: np.float64, y_mean: np.float64):

        if model['name'] != SVI_model_format.heterosc['name']:
            predictive = Predictive(model=model['model'], guide=guide, num_samples=1000, return_sites=("alpha", "beta", "sigma"))
            samples = predictive(X_train_torch, y_train_torch)

            alpha_samples = samples["alpha"].detach().numpy()
            beta_samples = samples["beta"].detach().numpy()

            y_hat = np.mean(np.exp(alpha_samples.T + np.dot(X_test, beta_samples[:,0].T)), axis=1)

        else:
            predictive = Predictive(model=model['model'], guide=guide, num_samples=1000,
                        return_sites=("alpha_mu", "beta_mu", "alpha_v", "beta_v"))
            samples = predictive(X_train_torch, y_train_torch)

            alpha_samples = samples["alpha_mu"].detach().numpy()
            beta_samples = samples["beta_mu"].detach().numpy()

            y_hat = np.mean(alpha_samples.T + np.dot(X_test, beta_samples[:,0].T), axis=1)

        # convert back to the original scale
        preds = y_hat # no need to do any conversion here because the Poisson model received untransformed y's
        y_true = y_test * y_std + y_mean

        return preds, y_true

def svi_main(data, component, regression_type, steps=1000, threshold=None):

    folder_path = os.path.join('..',root_dir,'reports','stats')
    file_path = os.path.join(folder_path,'regression_stats.csv')
                             
    svi_regression = SVI_regression_model(data, component)
    svi_dataset = svi_regression.get_data_for_component()

    if regression_type == 'poisson':
        model = SVI_model_format.poisson
    elif regression_type == 'linear':
        model = SVI_model_format.linear
    else:
        model = SVI_model_format.heterosc

    print(component)

    y, X, X_train_torch_, y_train_torch_, X_test, y_test, X_train, y_train, y_std, y_mean = svi_regression.preprocess(X_init=svi_dataset, model=model['name'])
    
    svi_guide = svi_regression.pyro_inference(X_train_torch=X_train_torch_, y_train_torch=y_train_torch_, model=model['model'], steps=steps)
    
    _preds, _y_true = svi_regression.post_process(guide=svi_guide, model=model, X_train_torch=X_train_torch_, y_train_torch=y_train_torch_, X_test=X_test, y_test=y_test, y_std=y_std, y_mean=y_mean)
    
    corr, mae, rae, rmse, r2, svi_trues, svi_pred = compute_error(trues=_y_true, predicted=_preds, threshold=threshold)
    print("CorrCoef: %.3f\nMAE: %.3f\nRMSE: %.3f\nR2: %.3f" % (corr, mae, rmse, r2))
    
    header =  ["regression_type", "component", "CorrCoef", "MAE", "RMSE", "R2"]

    results = [  regression_type,   component,       corr,   mae,   rmse,   r2]

    # Check if folder exists, create it if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created")

    # Check if file exists, write or append content accordingly
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(header)
            csv_writer.writerow(results)
    else:
        with open(file_path, 'a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(results)
            print("Content appended to file")

