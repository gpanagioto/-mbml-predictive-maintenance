import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from models import compute_error
from pyro.optim import Adam, ClippedAdam
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
from enum import Enum

class SVI_model_format():
    poisson = "POISSON"
    heterosc = "HETEROSCEDASTIC"
    linear = "LINEAR"

class SVI_regression_model():

    def __init__(self, data, component) -> None:
        
        self.data = data
        self.component = component

    def get_data_for_component(self):
    
        cols = ['voltmean_3h', 'rotatemean_3h',
                'pressuremean_3h', 'vibrationmean_3h', 'voltsd_3h', 'rotatesd_3h',
                'pressuresd_3h', 'vibrationsd_3h', 'voltmean_24h', 'rotatemean_24h',
                'pressuremean_24h', 'vibrationmean_24h', 'voltsd_24h', 'rotatesd_24h',
                'pressuresd_24h', 'vibrationsd_24h', 'error1count', 'error2count',
                'error3count', 'error4count', 'error5count','age',
                'model_model1', 'model_model2', 'model_model3', 'model_model4', self.component]

        return self.data[cols]
    
    def preprocess(self, X_init, format):

        X = X_init.to_numpy()

        # Keep the last column as target y
        y = X[:,-1]
        X = X[:,:-1]

        # standardize input features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X = (X - X_mean) / X_std
        print(X.shape)

        # standardize pickups
        y_mean = y.mean()
        y_std = y.std()
        y = (y - y_mean) / y_std
        print(y.shape)

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

        if format == SVI_model_format.poisson:
            y_train_torch = torch.tensor(y_train * y_std + y_mean).int()
        else:
            y_train_torch = torch.tensor(y_train).float()
           
        return y, X, X_train_torch, y_train_torch, X_test, y_test, X_train, y_train, y_std, y_mean
    
    def pyro_inference(self,X_train_torch, y_train_torch, model):
        # Define guide function
        guide = AutoDiagonalNormal(model)

        # Reset parameter values
        pyro.clear_param_store()

        # Define the number of optimization steps
        n_steps = 40000

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


        def post_process(self, guide, model, X_test, y_test, y_std, y_mean):
            predictive = Predictive(model, guide=guide, num_samples=1000, return_sites=("alpha", "beta", "sigma"))
            samples = predictive(X_train_torch, y_train_torch)

            alpha_samples = samples["alpha"].detach().numpy()
            beta_samples = samples["beta"].detach().numpy()
            y_hat = np.mean(np.exp(alpha_samples.T + np.dot(X_test, beta_samples[:,0].T)), axis=1)

            # convert back to the original scale
            preds = y_hat # no need to do any conversion here because the Poisson model received untransformed y's
            y_true = y_test * y_std + y_mean

            return preds, y_true, preds
        