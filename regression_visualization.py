import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()


##
def plotDist(y_true, y_pred=None, bins=10, norm_hist=False, histtype='bar', result_savepath='./nn_hist.png', model_name='Neural Network'):

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    # return an axe

    if y_pred is None:
        ax1.hist(y_true, bins=bins, histtype=histtype, density=norm_hist, color='b', alpha=0.5, label='True')
    else:
        ax1.hist([y_true, y_pred], bins=bins, histtype=histtype, density=norm_hist, color=['b', 'g'], alpha=0.5,
                 label=['True', 'Prediction ({})'.format(model_name)])

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Histogram')
    ax1.legend()

    fig.show()
    if result_savepath is not None:
        fig.savefig(result_savepath, bbox_inches='tight')
    return None


def plotTruePredScatter(y_true, y_pred, r2, result_savepath='./nn_scatter.png', model_name='Neural Network'):
    # plot the roc curve for the model
    max_y = max(y_true.max(), y_pred.max()) + max(np.std(y_true), np.std(y_pred))
    min_y = min(y_true.min(), y_pred.min()) - max(np.std(y_true), np.std(y_pred))

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot([min_y, max_y], [min_y, max_y], linestyle='--')
    ax1.scatter(y_true, y_pred, marker='o', cmap='Blues', alpha=0.5, label=model_name)
    ax1.set_xlim(min_y, max_y)
    ax1.set_ylim(min_y, max_y)
    ax1.set_aspect('equal', adjustable='box')
    # axis labels
    ax1.set_xlabel('True')
    ax1.set_ylabel('Prediction')
    ax1.legend(loc='lower right')

    if r2 is not None:
        ax1.text(0, 1, 'r2 = {:3f}'.format(r2),
                 horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)

    fig.show()
    if result_savepath is not None:
        fig.savefig(result_savepath, bbox_inches='tight')
    return None


if __name__ == '__main__':
    pass
