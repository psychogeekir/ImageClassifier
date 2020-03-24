import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score


import matplotlib.pyplot as plt

# use seaborn plotting defaults
import seaborn as sns
sns.set()

import torch
from torchvision.utils import save_image, make_grid


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

    topo = alldata.iloc[:, 0:img_size ** 2].values
    label = alldata.iloc[:, img_size ** 2 + 1].values
    classes = ('success', 'failure')  # 0: success, 1: failure

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
    # when class_weight is set to 'balanced', it will automatically set weight for imbalanced data
    # the highe value of gamma can lead to overfitting, check the ratio of  number of support vectors of number of all training samples
    svc = SVC(kernel='rbf', class_weight='balanced', gamma='scale')

    # make_pipeline() can assemble procedures
    model = make_pipeline(svc)
    # model = make_pipeline(pca, svc)

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

    Xtrain = traindata_oversampling.iloc[:, :img_size ** 2].values
    ytrain = traindata_oversampling['label'].values

    ## hyperparamter tuning use GridSearch

    # C in svc is the regularization coefficient, it can be combined with class_weight to handle imbalanced dataset
    # 'svc_C' is a predefined format, if LinearSVC is used, 'linearsvc_C'
    param_grid = {'svc__C': [230, 235, 236, 237, 238]}
    # 5 times cross validation, train set will be further divided into development set and train set
    grid = GridSearchCV(model, param_grid, scoring='f1_macro', iid='deprecated', cv=5, n_jobs=-1)

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

    print('balanced_accuracy_score is {}'.format(balanced_accuracy_score(ytest, ypred)))

    ## plot confusion matrix

    sns.set(font_scale=1.4)
    cm, tn, fp, fn, tp = plotConfusionMatrix(ytest, ypred, classes, labels=(0, 1),
                                             result_savepath='./svm_confusion_matrix.png')


    ## in binary classification problem, the decision function of SVM outputs the distance of sample point to the decision hyperplane
    # the sign represents the labels, - -> 0, + -> 1
    score_test = model.steps[0][1].decision_function(Xtest)
    score_margin = model.steps[0][1].decision_function(model.steps[0][1].support_vectors_)

    ytest[score_test < np.min(score_margin)]  # should be 0
    ytest[score_test > np.max(score_margin)]  # should be 1

    x = Xtest[score_test >= np.min(score_margin), :]