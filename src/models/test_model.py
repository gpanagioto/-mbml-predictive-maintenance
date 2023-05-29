# Define guide function
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
import pyro
from pyro.infer import Predictive
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

# Make predictions for test set
def test_nn(model,guide,X_test_torch):
    # make predictions for test set using the trained model
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000,return_sites=("obs", "_RETURN"))
    samples = predictive(X_test_torch)
    y_pred = samples["obs"].mean(axis=0).detach().numpy()
    return y_pred

def test_nn_beta(model,guide,X_test_torch):
    # make predictions for test set using the trained model
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000,return_sites=("beta",))
    samples = predictive(X_test_torch)
    print("Estimated beta:", samples["beta"].mean(axis=0).detach().numpy())

def test_nn_c(model,guide,X_test_torch,thres):
    # make predictions for test set using the trained model
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=2000,return_sites=("obs", "_RETURN"))
    samples = predictive(X_test_torch)
    y_pred = samples["obs"].mean(axis=0).detach().numpy()
    #threshold predictions
    y_pred[y_pred<=thres]=0
    y_pred[y_pred>thres]=1
    return y_pred

def test_lg_c(logreg,X_test_torch):
    # make predictions for test set using the sklearn model
    y_hat = logreg.predict(X_test_torch)
    return y_hat



def test_model_c(alpha_hat,beta_hat,X_test_torch,thres=None):
    # make predictions for test set either using threshold or argmax
    y_hat = alpha_hat + np.dot(X_test_torch, beta_hat)
    if thres:
        thres=0.01
        y_hat[y_hat<=thres]=0
        y_hat[y_hat>thres]=1
    else: 
        y_hat = np.argmax(y_hat, axis=1)
    return y_hat




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





def test_c(y_hat,y_test):
    # evaluate prediction accuracy
    print("Accuracy:", 1.0*np.sum(y_hat == y_test) / len(y_test))
# function to evaluate predictions

def evaluate(y_test, y_hat):
    # calculate and display confusion matrix
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_hat, labels=labels)
    print('Confusion matrix\n- x-axis is true labels (no failure, failure)\n- y-axis is predicted labels')
    print(cm)
    # calculate precision, recall, and F1 score
    accuracy = float(np.trace(cm)) / np.sum(cm)
    precision = precision_score(y_test, y_hat, average=None, labels=labels)[1]
    recall = recall_score(y_test, y_hat, average=None, labels=labels)[1]
    f1 = 2 * precision * recall / (precision + recall)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1 score:", f1)
