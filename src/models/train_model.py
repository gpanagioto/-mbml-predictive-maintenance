# Define guide function
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from src.models.models import FFNN_interpretable as modelFFNN_interpretable
from src.models.models import FFNN as modelFFNN
from pyro.infer import Predictive

def train_nn(model0, X_train_torch, y_train_torch):
    if model0 == modelFFNN:
        model = model0(n_in=X_train_torch.shape[1], n_hidden=32, n_out=1)
    elif model0 == modelFFNN_interpretable:
        model = model0(n_in=X_train_torch.shape[1]-1, n_hidden=32, n_out=1)
    guide = AutoDiagonalNormal(model)
    pyro.clear_param_store()
    # Define the number of optimization steps
    n_steps = 10000

    # Setup the optimizer
    adam_params = {"lr": 0.01}
    optimizer = Adam(adam_params)

    # Setup the inference algorithm
    elbo = Trace_ELBO(num_particles=1)
    svi = SVI(model, guide, optimizer, loss=elbo)

    # Do gradient steps
    for step in range(n_steps):
        elbo = svi.step(X_train_torch,y_train_torch)
        if step % 500 == 0:
            print("[%d] ELBO: %.1f" % (step, elbo))
    return model,guide


