import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, RandomizedSearchCV

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
    n_cpu = 8

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

    # pca = RandomizedPCA(n_components=20, whiten=True, random_state=manualSeed)
    # when class_weight is set to 'balanced', it will automatically set weight for imbalanced data
    rfc = RandomForestClassifier(criterion='gini', oob_score=False, class_weight='balanced')

    # make_pipeline() can assemble procedures
    model = make_pipeline(rfc)
    # model = make_pipeline(pca, rfc)

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

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=100)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    param_grid = {'randomforestclassifier__n_estimators': n_estimators,
                  'randomforestclassifier__max_features': max_features,
                  'randomforestclassifier__max_depth': max_depth,
                  'randomforestclassifier__min_samples_split': min_samples_split,
                  'randomforestclassifier__min_samples_leaf': min_samples_leaf,
                  'randomforestclassifier__bootstrap': bootstrap}

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

    print('balanced_accuracy_score is {}'.format(balanced_accuracy_score(ytest, ypred)))

    ## plot confusion matrix

    sns.set(font_scale=1.4)
    cm, tn, fp, fn, tp = plotConfusionMatrix(ytest, ypred, classes, labels=(0, 1),
                                             result_savepath='./rf_confusion_matrix.png')



