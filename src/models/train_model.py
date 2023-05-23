# Define guide function
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from src.models.models import FFNN, FFNN_interpretable, RNN, FFNN_c
from pyro.infer import Predictive
from sklearn import datasets, linear_model



def train_nn(model0, X_train_torch, y_train_torch):
    if model0 =="modelFFNN":
        model = FFNN(n_in=X_train_torch.shape[1], n_hidden=32, n_out=1)
    if model0 == "modelFFNN_c":
        model = FFNN_c(n_in=X_train_torch.shape[1], n_hidden=32, n_out=1)
    if model0 == "modelRNN":
        model = RNN(n_in=X_train_torch.shape[1], n_hidden=32, n_out=1)
    elif model0 == "modelFFNN_interpretable":
        model = FFNN_interpretable(n_in=X_train_torch.shape[1]-1, n_hidden=32, n_out=1)
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



def lg_c(X_train_torch, y_train_torch):
    # create and fit logistic regression model
    logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', C=1)
    logreg.fit(X_train_torch, y_train_torch)
    return logreg

#model using SVI
def train_c_svi(model, X_train, y_train, n_cat):
        # Define guide function
    guide = AutoMultivariateNormal(model)

    # Reset parameter values
    pyro.clear_param_store()

    # Define the number of optimization steps
    n_steps = 40000

    # Setup the optimizer
    adam_params = {"lr": 0.001}
    optimizer = ClippedAdam(adam_params)

    # Setup the inference algorithm
    elbo = Trace_ELBO(num_particles=1)
    svi = SVI(model, guide, optimizer, loss=elbo)

    # Do gradient steps
    for step in range(n_steps):
        elbo = svi.step(X_train, n_cat, y_train)#y_train had -1
        if step % 1000 == 0:
            print("[%d] ELBO: %.1f" % (step, elbo))
    predictive = Predictive(model, guide=guide, num_samples=1000,return_sites=("alpha", "beta"))
    samples = predictive(X_train, n_cat,y_train)
    alpha_hat = samples["alpha"].detach().squeeze().mean(axis=0).numpy()
    beta_hat = samples["beta"].detach().squeeze().mean(axis=0).numpy()
    return model,guide, alpha_hat, beta_hat



def train_c_mcmc(model, X_train, y_train, n_cat):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100, num_chains=1)
    mcmc.run(X_train, n_cat, y_train) # Pyro accepts categories starting from 0
    samples = mcmc.get_samples()
    alpha_hat = samples["alpha"].detach().squeeze().mean(axis=0).numpy()
    beta_hat = samples["beta"].detach().squeeze().mean(axis=0).numpy()
    return mcmc, alpha_hat, beta_hat


