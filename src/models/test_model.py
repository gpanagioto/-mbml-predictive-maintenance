# Define guide function
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
import pyro
from pyro.infer import Predictive
import numpy as np
import matplotlib.pyplot as plt

# Make predictions for test set
def test_nn(model,guide,X_test_torch):
    # Predict
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000,return_sites=("obs", "_RETURN"))
    samples = predictive(X_test_torch)
    y_pred = samples["obs"].mean(axis=0).detach().numpy()
    return y_pred

def test_nn_beta(model,guide,X_test_torch):
    # Predict
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000,return_sites=("beta",))
    samples = predictive(X_test_torch)
    print("Estimated beta:", samples["beta"].mean(axis=0).detach().numpy())

def test_nn_c(model,guide,X_test_torch):
    # Predict
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000,return_sites=("obs", "_RETURN"))
    samples = predictive(X_test_torch)
    y_pred = samples["obs"].mean(axis=0).detach().numpy()
    y_pred[y_pred<=0.5]=0
    y_pred[y_pred>0.5]=1
    return y_pred

def test_lg_c(logreg,X_test_torch):
    # make predictions for test set
    y_hat = logreg.predict(X_test_torch)
    return y_hat



def test_model_c(alpha_hat,beta_hat,X_test_torch):
    # make predictions for test set
    y_hat = alpha_hat + np.dot(X_test_torch, beta_hat)
    y_hat = np.argmax(y_hat, axis=1)
    return y_hat


def test_c(y_hat,y_test):
    print("predictions:", y_hat)
    print("true values:", y_test)
    # evaluate prediction accuracy
    print("Accuracy:", 1.0*np.sum(y_hat == y_test) / len(y_test))

def mae_test(y_pred,y_test):
    mae = np.mean(np.abs(y_pred - y_test))
    print("MAE:", mae)
    return 

def plot_pred(y_pred,y_test,y_std,y_mean,threshold,start=None,end=None):
    fig = plt.figure(figsize=(20, 10))

    start = start
    end = end
    y_true = y_test * y_std + y_mean
    y_pre = y_pred * y_std + y_mean

    plt.plot(y_true[y_pre>threshold], 'r.-', label='test')
    plt.plot(y_pre[y_pre>threshold], 'b-', label='pred')
    plt.legend()
    plt.show()
    return y_true,y_pre



