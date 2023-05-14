# Define guide function
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
import pyro
from pyro.infer import Predictive
import numpy as np

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


def mae_test(y_pred,y_test):
    mae = np.mean(np.abs(y_pred - y_test))
    print("MAE:", mae)
    return 

