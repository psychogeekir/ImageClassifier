import numpy as np
import pandas as pd


from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


import seaborn as sns
sns.set()

import torch
from torchvision.utils import save_image, make_grid


def plotConfusionMatrix(cm, classes=('sucess', 'failure'), normalize=True, result_savepath='./nn_confusion_matrix.png'):

    nrows, ncols = cm.shape
    annot = np.empty_like(cm).astype(str)

    if normalize:
        cm_sum = np.sum(cm, axis=1, keepdims=True)  # sum each actual class, recall
        cm_perc = cm / cm_sum * 100

        if nrows + ncols == 4:  # binary
            names = ('TN', 'FP', 'FN', 'TP')
            k = 0
            for i in range(nrows):
                for j in range(ncols):
                    c = cm[i, j]
                    p = cm_perc[i, j]
                    if c == 0:
                        annot[i, j] = ''
                    else:
                        annot[i, j] = '{}: {:.2f}%'.format(names[k], p)
                    k += 1
        else:
            for i in range(nrows):
                for j in range(ncols):
                    c = cm[i, j]
                    p = cm_perc[i, j]
                    if c == 0:
                        annot[i, j] = ''
                    else:
                        annot[i, j] = '{:.2f}%'.format(p)
    else:
        if nrows + ncols == 4:  # binary
            names = ('TN', 'FP', 'FN', 'TP')
            k = 0
            for i in range(nrows):
                for j in range(ncols):
                    c = cm[i, j]
                    if c == 0:
                        annot[i, j] = ''
                    else:
                        annot[i, j] = '{}: {:d}'.format(names[k], c)
                    k += 1
        else:
            for i in range(nrows):
                for j in range(ncols):
                    c = cm[i, j]
                    if c == 0:
                        annot[i, j] = ''
                    else:
                        annot[i, j] = '{:d}%'.format(c)


    cm = pd.DataFrame(cm, index=classes, columns=classes)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)

    sns_plot = sns.heatmap(cm, ax=ax1, square=False, annot=annot, fmt='',
                           cmap='Blues', cbar=True, annot_kws={"size": 16})

    # ax1.set_ylim([0, len(labels)])  # matplotlib can broke the position of labels
    fig.show()

    if result_savepath is not None:
        sns_plot.figure.savefig(result_savepath, bbox_inches='tight')
    return None


def plotROCCurve(fpr, tpr, close_default_clf=None, result_savepath='./nn_ROC.png', model_name='Neural Network'):
    # plot the roc curve for the model
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    ax1.plot(fpr, tpr, marker='.', label=model_name)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal', adjustable='box')
    # axis labels
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate (Recall)')
    ax1.legend()

    if close_default_clf is not None:
        ax1.plot(fpr[close_default_clf], tpr[close_default_clf], '^', c='k', markersize=15)

    fig.show()
    if result_savepath is not None:
        fig.savefig(result_savepath, bbox_inches='tight')
    return None


def markThresholdOnROC(t, thresholds, fpr, tpr, result_savepath='./nn_ROC_with_threshold.png', model_name='Neural Network'):
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plotROCCurve(fpr, tpr, close_default_clf=close_default_clf, result_savepath=result_savepath, model_name=model_name)
    return None


def plotPRCurve(precision, recall, no_skill, close_default_clf=None, result_savepath='./nn_PR.png', model_name='Neural Network'):
    # plot the roc curve for the model
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax1.plot(recall, precision, marker='.', label=model_name)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal', adjustable='box')
    # axis labels
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.legend()

    if close_default_clf is not None:
        ax1.plot(recall[close_default_clf], precision[close_default_clf], '^', c='k', markersize=15)

    fig.show()
    if result_savepath is not None:
        fig.savefig(result_savepath, bbox_inches='tight')
    return None


def markThresholdOnPRCurve(t, thresholds, precision, recall, no_skill, result_savepath='./nn_PR_with_threshold.png', model_name='Neural Network'):
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plotPRCurve(precision, recall, no_skill, close_default_clf=close_default_clf, result_savepath=result_savepath, model_name=model_name)
    return None


if __name__ == '__main__':
    array = [[13, 1, 1, 0, 2, 0],
             [3, 9, 6, 0, 1, 0],
             [0, 0, 16, 2, 0, 0],
             [0, 0, 0, 13, 0, 0],
             [0, 0, 0, 0, 15, 0],
             [0, 0, 1, 0, 0, 15]]
    a = np.array(array)

    df_cm = pd.DataFrame(a, index=range(6), columns=range(6))
    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16})  # font size
    plt.show()

    plotConfusionMatrix(a, classes=('a', 'b', 'c', 'd', 'e', 'f'), normalize=True, result_savepath=None)

    array = [[13, 1],
             [3, 9]]

    b = np.array(array)

    plotConfusionMatrix(b, classes=('sucess', 'failure'), normalize=False,
                        result_savepath=None)
