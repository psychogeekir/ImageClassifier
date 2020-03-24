import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

import torch
from torchvision.utils import save_image, make_grid

##


def classification_metrics(y_true, y_pred, y_prob, y_score, metric_names, name_funcs_dict=None):
    # use sklearn.metrics to compute metrics
    # y_pred and y_true are nd arrays

    # y_score -> probability transform like sigmoid() -> y_prob -> y_pred (labels)

    # return scores, the larger, the better

    # y_true --- True labels or binary label indicators.
    # The binary and multiclass cases expect labels with shape (n_samples,)
    # while the multilabel case expects binary label indicators with shape (n_samples, n_classes).

    # y_pred --- Predicted labels, as returned by a classifier.

    # y_prob --- Predicted probabilities, as returned by a classifier’s predict_proba method.
    # If y_pred.shape = (n_samples,) the probabilities provided are assumed to be that of the positive class.
    # The labels in y_pred are assumed to be ordered alphabetically, as done by preprocessing.LabelBinarizer

    # y_score --- Target scores. In the binary and multilabel cases, these can be either probability estimates or
    # non-thresholded decision values (as returned by decision_function on some classifiers, e.g., SVM).
    # In the multiclass case, these must be probability estimates which sum to 1.
    # The binary case expects a shape (n_samples,), and the scores must be the scores of the class with the greater label.
    # The multiclass and multilabel cases expect a shape (n_samples, n_classes).
    # In the multiclass case, the order of the class scores must correspond to the order of labels, if provided,
    # or else to the numerical or lexicographical order of the labels in y_true.

    if name_funcs_dict is None:
        name_funcs_dict = {

            'accuracy': metrics.accuracy_score,
            'balanced_accuracy': metrics.balanced_accuracy_score,

            # average will be ignored when y_true is binary.
            'average_precision_macro': lambda y_true, y_score: metrics.average_precision_score(y_true, y_score, average='macro', pos_label=1),
            'average_precision_micro': lambda y_true, y_score: metrics.average_precision_score(y_true, y_score, average='micro', pos_label=1),
            'average_precision_weighted': lambda y_true, y_score: metrics.average_precision_score(y_true, y_score, average='weighted', pos_label=1),
            'average_precision_samples': lambda y_true, y_score: metrics.average_precision_score(y_true, y_score, average='samples', pos_label=1),

            # The smaller the Brier score, the better, we add a minus sign
            'neg_brier_score': lambda y_true, y_prob: -metrics.brier_score_loss(y_true, y_prob, pos_label=1),

            # The smaller the log loss, the better, we add a minus sign
            'neg_log_loss': lambda y_true, y_prob: -metrics.log_loss(y_true, y_prob, labels=[0, 1]),

            'f1_binary': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary'),
            'f1_micro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, labels=None, average='micro'),
            'f1_macro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, labels=None, average='macro'),
            'f1_weighted': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, labels=None, average='weighted'),
            'f1_samples': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, labels=None, average='samples'),

            'precision_binary': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary'),
            'precision_micro': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, labels=None, average='micro'),
            'precision_macro': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, labels=None, average='macro'),
            'precision_weighted': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, labels=None, average='weighted'),
            'precision_samples': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, labels=None, average='samples'),

            'recall_binary': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary'),
            'recall_micro': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, labels=None, average='micro'),
            'recall_macro': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, labels=None, average='macro'),
            'recall_weighted': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, labels=None, average='weighted'),
            'recall_samples': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, labels=None, average='samples'),

            'jaccard_binary': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, labels=None, pos_label=1, average='binary'),
            'jaccard_micro': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, labels=None, average='micro'),
            'jaccard_macro': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, labels=None, average='macro'),
            'jaccard_weighted': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, labels=None, average='weighted'),
            'jaccard_samples': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, labels=None, average='samples'),

            # average will be ignored when y_true is binary
            # multi_class is only for multi-class problem
            # 'ovr' is sensitive to class imbalance even when average == 'macro', because class imbalance affects the composition of each of the ‘rest’ groupings.
            'roc_auc_binary': lambda y_true, y_score: metrics.roc_auc_score(y_true, y_score, multi_class='raise', labels=[0, 1]),
            'roc_auc_ovr_macro': lambda y_true, y_score: metrics.roc_auc_score(y_true, y_score, multi_class='ovr', labels=[0, 1], average='macro'),
            'roc_auc_ovo_macro': lambda y_true, y_score: metrics.roc_auc_score(y_true, y_score, multi_class='ovo', labels=[0, 1], average='macro'),
            'roc_auc_ovr_micro': lambda y_true, y_score: metrics.roc_auc_score(y_true, y_score, multi_class='ovr', labels=[0, 1], average='micro'),
            'roc_auc_ovo_micro': lambda y_true, y_score: metrics.roc_auc_score(y_true, y_score, multi_class='ovo', labels=[0, 1], average='micro'),
            'roc_auc_ovr_samples': lambda y_true, y_score: metrics.roc_auc_score(y_true, y_score, multi_class='ovr', labels=[0, 1], average='samples'),
            'roc_auc_ovo_samples': lambda y_true, y_score: metrics.roc_auc_score(y_true, y_score, multi_class='ovo', labels=[0, 1], average='samples'),
            'roc_auc_ovr_weighted': lambda y_true, y_score: metrics.roc_auc_score(y_true, y_score, multi_class='ovr', labels=[0, 1], average='weighted'),
            'roc_auc_ovo_weighted': lambda y_true, y_score: metrics.roc_auc_score(y_true, y_score, multi_class='ovo', labels=[0, 1], average='weighted'),
        }

    scores = {}
    for metric_name in metric_names:
        if metric_name in ('neg_brier_score', 'neg_log_loss'):
            scores[metric_name] = name_funcs_dict[metric_name](y_true, y_prob)
        elif 'roc_auc' in metric_name or 'average_precision' in metric_name:
            scores[metric_name] = name_funcs_dict[metric_name](y_true, y_score)
        else:
            scores[metric_name] = name_funcs_dict[metric_name](y_true, y_pred)

    return scores


def plotConfusionMatrix(ytest, ypred, classes=('sucess', 'failure'), labels=(0, 1), result_savepath='./nn_confusion_matrix.png'):
    mat = confusion_matrix(ytest, ypred, labels=labels)
    tn, fp, fn, tp = mat.ravel()
    print('number of True Negative: {}'.format(tn))
    print('number of False Positive: {}'.format(fp))
    print('number of False Negative: {}'.format(fn))
    print('number of True Positive: {}'.format(tp))

    cm = pd.DataFrame(mat, index=classes, columns=classes)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    sns_plot = sns.heatmap(cm, ax=ax1, square=False, annot=True, fmt='d',
                           cmap='Blues', cbar=True, annot_kws={"size": 16})
    # ax1.set_ylim([0, 2])
    fig.show()
    sns_plot.figure.savefig(result_savepath, bbox_inches='tight')
    return cm, tn, fp, fn, tp


if __name__ == '__main__':
    ## read data

    alldata = pd.read_csv('./cleandata.csv', header=0, index_col=None)
    img_size = 10
    nc = 1

    manualSeed = 42
    n_cpu = 8

    topo = alldata.iloc[:, 0:img_size**2].values
    label = alldata.iloc[:, img_size**2+1].values
    classes = ('success', 'failure')  # 0: success, 1: failure

    metric_names = ('accuracy', 'f1_binary')
    name_funcs_dict = None

    ## plot some samples

    nsamples = 100
    topo_array = alldata.iloc[:nsamples, 0:img_size**2].values
    topo_tensor = np.reshape(topo_array, (nsamples, nc, img_size, img_size), order='C')
    topo_tensor = torch.from_numpy(topo_tensor).type(torch.FloatTensor)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(topo_tensor[:64], padding=5, pad_value=1, normalize=True).numpy(), (1, 2, 0)))
    plt.show()

    ## build the pipeline

    pca = RandomizedPCA(n_components=20, whiten=True, random_state=manualSeed)
    mlpc = MLPClassifier(solver='adam', max_iter=1000, shuffle=True, early_stopping=False,
                         validation_fraction=0.2, beta_1=0.9, beta_2=0.999, n_iter_no_change=10,
                         verbose=True, random_state=manualSeed)

    # make_pipeline() can assemble procedures
    model = make_pipeline(mlpc)
    # model = make_pipeline(pca, mlpc)

    ## split dataset
    Xtrain, Xtest, ytrain, ytest = train_test_split(topo, label, test_size=0.2, random_state=manualSeed)

    ## oversampling
    train_data = pd.DataFrame(Xtrain)
    train_data['label'] = ytrain

    class_sample_counts = np.bincount(train_data['label'])
    n_failure = class_sample_counts[1]
    n_success = class_sample_counts[0]
    moredata = train_data.loc[train_data['label'] == 1, :].sample(n=n_success - n_failure,
                                                                  replace=True, random_state=manualSeed)

    traindata_oversampling = train_data.append(moredata, ignore_index=True)
    print('oversampling --- n_failure: {}, n_success: {}'.format(sum(traindata_oversampling['label'] == 1),
                                                                 sum(traindata_oversampling['label'] == 0)))

    Xtrain = traindata_oversampling.iloc[:, :img_size**2].values
    ytrain = traindata_oversampling['label'].values


    ## hyperparamter tuning use GridSearch

    # hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
    hidden_layer_sizes = [(80, 60, 40, 20), (80, 40, 20)]

    # Activation function for the hidden layer
    activation = ['relu', 'tanh', 'logistic']

    # L2 penalty (regularization term) parameter
    alpha = [0.00008, 0.0001, 0.0002]

    # Size of minibatches for stochastic optimizers
    batch_size = [int(x) for x in np.linspace(400, 1500, num=11)]

    # learning rate, only used for SGD solver
    # learning_rate = ['constant', 'adaptive', 'invscaling']

    # initial learning rate, only used for SGD or Adam solvers
    learning_rate_init = [0.002, 0.005, 0.006, 0.008, 0.01]

    param_grid = {
        'mlpclassifier__hidden_layer_sizes': hidden_layer_sizes,
        'mlpclassifier__activation': activation,
        'mlpclassifier__alpha': alpha,
        'mlpclassifier__batch_size': batch_size,
        'mlpclassifier__learning_rate_init': learning_rate_init,
    }

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='f1_macro',
                              n_iter=100, cv=3, verbose=2, random_state=manualSeed, n_jobs=-1)

    grid.fit(Xtrain, ytrain)
    print(grid.best_params_)

    ## use the best hyperparameter to build the model and make predictions
    model = grid.best_estimator_
    ypred = model.predict(Xtest)

    ## visualize the prediction
    fig, ax = plt.subplots(4, 6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(Xtest[i].reshape(img_size, img_size), cmap='gray')
        axi.set(xticks=[], yticks=[])
        axi.set_ylabel(classes[ypred[i]].split()[-1],
                       color='black' if ypred[i] == ytest[i] else 'red')
    fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
    fig.show()

    ## print classification_report

    print(classification_report(ytest, ypred, labels=[0, 1], target_names=classes, digits=3))

    scores = classification_metrics(ytest, ypred, y_prob=None, y_score=None, metric_names=metric_names, name_funcs_dict=None)
    print('metrics--- ' + ' '.join('{}: {:.3f}'.format(metric_name, scores[metric_name]) for metric_name in metric_names))


    ## plot confusion matrix

    sns.set(font_scale=1.4)
    cm, tn, fp, fn, tp = plotConfusionMatrix(ytest, ypred, classes, labels=(0, 1),
                                             result_savepath='./mlpc_confusion_matrix.png')

