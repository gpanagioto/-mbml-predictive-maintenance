import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os

def true_vs_preds_plot(y:np.array, predictions:np.array, window:List, fig_path:str, name:str):

    fig = plt.figure(figsize=(20,10))

    if window:
        start = window[0]
        end = window[1]
    else:
        pass

    plt.plot(y[np.where(predictions>=0)], 'r-', label = 'test')
    plt.plot(predictions[np.where(predictions>=0)], 'b-', label = 'pred')

    plt.legend()

    if fig_path:
        try:
            plt.savefig(os.path.join(fig_path,name))
        except:
            print(Exception)
            pass

