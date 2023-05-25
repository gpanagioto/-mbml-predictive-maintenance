import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from src.models.models import poisson_model, linear_model, heteroscedastic_model 

class MCMC_model_format():
    '''
    defining the type of the regression
    '''
    poisson = {'name':"POISSON", 'model':poisson_model}
    heterosc = {'name':"HETEROSCEDASTIC", 'model':heteroscedastic_model}
    linear = {'name':"LINEAR", 'model':linear_model}

class MCMC_regression_model():

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
    
    def preprocess(self, X_init):

        X_init = X_init.to_numpy()

        y = X_init[:,-1]
        X = X_init[:,:-1]

        X_train_unsc, X_test_unsc, y_train_unsc, y_test_unsc = train_test_split(X, y, test_size=0.2, random_state=0)

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

        return y, X, X_train_torch, y_train_torch, X_test, y_test, X_train, y_train, y_std, y_mean
    

    def pyro_inference(self, X_train_torch, y_train_torch, model, num_samples):

        # Run inference in Pyro
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=200, num_chains=1)
        mcmc.run(X_train_torch, y_train_torch)

        # Show summary of inference results
        print(mcmc.summary())

        return mcmc.get_samples()

    def post_process(self, X_test, X_train, samples, y_std, y_mean, y_test):

        posterior_samples = samples

        # Compute predictions
        y_hat = np.mean(posterior_samples["alpha"].numpy().T + np.dot(X_test, posterior_samples["beta"].numpy().T), axis=1)
        y_hat_train = np.mean(posterior_samples["alpha"].numpy().T + np.dot(X_train, posterior_samples["beta"].numpy().T), axis=1)

        # Convert back to the original scale
        preds = y_hat * y_std + y_mean
        preds_train = y_hat_train * y_std + y_mean
        y_true = y_test * y_std + y_mean

        return preds, y_true