import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchvision.utils import save_image, make_grid

##
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics

##
from Classifier.classification_evaluation import confusionMatrix, classification_metrics, calPRAUC, adjust_threshold, \
    optimalBalancedThresholdOnROC, optimalBalancedThresholdOnPRCurve
from Classifier.classification_visualization import plotConfusionMatrix, plotPRCurve, plotROCCurve, \
    markThresholdOnPRCurve, markThresholdOnROC

##
def pickClassifier(mdlName, manualSeed):
    mdl_dict = {
        'SVC': SVC(kernel='rbf', class_weight='balanced', gamma='scale', probability=False),
        'RandomForestClassifier': RandomForestClassifier(criterion='gini', oob_score=False, class_weight='balanced'),
        'MLPClassifier': MLPClassifier(solver='adam', max_iter=1000, shuffle=True, early_stopping=False,
                                       validation_fraction=0.2, beta_1=0.9, beta_2=0.999, n_iter_no_change=10,
                                       verbose=True, random_state=manualSeed),
        'KNeighborsClassifier': KNeighborsClassifier(algorithm='auto',
                                                     metric='minkowski', metric_params=None,
                                                     n_jobs=-1),
    }
    return mdl_dict[mdlName]


def pickParamGrid(mdlName):
    param_dict = {

        'SVC': {
            'svc__C': [230, 235, 236, 237, 238],
        },

        'RandomForestClassifier': {
            'randomforestclassifier__n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=100)],
            'randomforestclassifier__max_features':  ['auto', 'sqrt'],
            'randomforestclassifier__max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
            'randomforestclassifier__min_samples_split': [2, 5, 10],
            'randomforestclassifier__min_samples_leaf': [1, 2, 4],
            'randomforestclassifier__bootstrap': [True, False],
        },

        'MLPClassifier': {
            'mlpclassifier__hidden_layer_sizes': [(80, 60, 40, 20), (80, 40, 20)],
            'mlpclassifier__activation': ['relu', 'tanh', 'logistic'],
            'mlpclassifier__alpha': [0.00008, 0.0001, 0.0002],
            'mlpclassifier__batch_size': [int(x) for x in np.linspace(400, 1500, num=11)],
            'mlpclassifier__learning_rate_init': [0.002, 0.005, 0.006, 0.008, 0.01],
        },

        'KNeighborsClassifier': {
            'kneighborsclassifier__n_neighbors': [7, 8, 9, 10],
            'kneighborsclassifier__weights': ['uniform', 'distance'],
            'kneighborsclassifier__leaf_size': [x for x in range(30, 50)],  #[int(x) for x in np.linspace(20, 100, num=20)],
            'kneighborsclassifier__p': [1, 2],
        },
    }
    return param_dict[mdlName]


##
def oversampling(Xtrain, ytrain):

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
    return Xtrain, ytrain



##
def plotBatch(topo, img_size, nc, nsamples=100):
    topo_array = topo[:nsamples, :]
    topo_tensor = np.reshape(topo_array, (nsamples, nc, img_size, img_size), order='C')
    topo_tensor = torch.from_numpy(topo_tensor).type(torch.FloatTensor)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(topo_tensor[:64], padding=5, pad_value=1, normalize=True).numpy(), (1, 2, 0)))
    plt.show()
    return None


def plotBatchPrediction(ytest, ypred, Xtest, img_size, classes):
    ## visualize the prediction
    fig, ax = plt.subplots(4, 6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(Xtest[i].reshape(img_size, img_size), cmap='gray')
        axi.set(xticks=[], yticks=[])
        axi.set_ylabel(classes[ypred[i]].split()[-1],
                       color='black' if ypred[i] == ytest[i] else 'red')
    fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
    fig.show()
    return None

##
if __name__ == '__main__':
    ## read data

    # mdlName = 'MLPClassifier'
    # mdlName = 'RandomForestClassifier'
    # mdlName = 'SVC'
    mdlName = 'KNeighborsClassifier'

    result_savepath = mdlName + '_result'
    os.makedirs(result_savepath, exist_ok=True)

    # refer to sklearn.metrics.make_scorer, the default input is y_true and y_pred
    # if y_prob is needed, wrap this with make_scorer
    scorers = {
        'precision_binary': metrics.make_scorer(lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary')),
        'recall_binary': metrics.make_scorer(lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary')),
        'f1_binary': metrics.make_scorer(lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary')),
        'pr_auc': metrics.make_scorer(lambda y_true, y_prob: calPRAUC(y_true, y_prob),
                                      needs_proba=(False if mdlName == 'SVC' else True),
                                      needs_threshold=(True if mdlName == 'SVC' else False)),
    }

    alldata = pd.read_csv('./cleandata.csv', header=0, index_col=None)
    img_size = 10
    nc = 1

    manualSeed = 42

    topo = alldata.iloc[:, 0:img_size ** 2].values
    label = alldata.iloc[:, img_size ** 2 + 1].values
    classes = ('success', 'failure')  # 0: success, 1: failure

    ## crop quarter

    plotBatch(topo, img_size, nc, nsamples=100)

    topo = np.reshape(topo, (topo.shape[0], nc, img_size, img_size))
    topo = topo[:, :, :int(img_size/2), int(img_size/2):]
    topo = np.reshape(topo, (topo.shape[0], -1))

    ## plot some samples
    plotBatch(topo, int(img_size/2), nc, nsamples=100)

    ## build the pipeline

    pca = RandomizedPCA(n_components=10, whiten=True, random_state=manualSeed)
    mdl = pickClassifier(mdlName, manualSeed)

    # make_pipeline() can assemble procedures
    model = make_pipeline(mdl)
    # model = make_pipeline(pca, mdl)

    ## split dataset
    Xtrain, Xtest, ytrain, ytest = train_test_split(topo, label,
                                                    test_size=0.2,
                                                    stratify=None,  # label,
                                                    random_state=manualSeed)

    ## oversampling
    # Xtrain, ytrain = oversampling(Xtrain, ytrain)

    ## hyperparamter tuning use RandomizedSearchCV

    param_grid = pickParamGrid(mdlName)

    # 5 times cross validation, train set will be further divided into development set and train set
    # grid = GridSearchCV(model, param_grid, scoring=scorers, refit='precision_binary', iid='deprecated', cv=5, n_jobs=-1)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              scoring=scorers, refit='f1_binary',
                              n_iter=100, cv=5, verbose=2, random_state=manualSeed, n_jobs=-1)

    grid.fit(Xtrain, ytrain)
    print(grid.best_params_)

    ## use the best hyperparameter to build the model and make predictions
    model = grid.best_estimator_

    ## fine tune decision threshold

    if mdlName == 'RandomForestClassifier' or mdlName == 'MLPClassifier' or mdlName == 'KNeighborsClassifier':
        yscore = model.predict_proba(Xtest)[:, 1]  # only need probability for class 1
    elif mdlName == 'SVC':
        yscore = model.decision_function(Xtest)

    ## plot ROC and PR curve
    # calculate roc and pr curves
    curves = classification_metrics(y_true=ytest, y_pred=None, y_prob=yscore, y_score=yscore,
                                    metric_names=('roc_curve', 'pr_curve', 'pr_auc'), name_funcs_dict=None)

    print('PR_AUC is {:.3f}'.format(curves['pr_auc']))

    # thresholds_roc[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
    fpr, tpr, thresholds_roc = curves['roc_curve']
    plotROCCurve(fpr, tpr, close_default_clf=None,
                 result_savepath=result_savepath + '/{}_ROC.png'.format(mdlName),
                 model_name=mdlName)

    # The last precision and recall values are 1. and 0. respectively
    # and do not have a corresponding threshold. This ensures that the graph starts on the y axis.
    precision, recall, thresholds_pr = curves['pr_curve']
    no_skill = len(ytest[ytest == 1]) / len(ytest)
    plotPRCurve(precision, recall, no_skill, close_default_clf=None,
                result_savepath=result_savepath + '/{}_PR.png'.format(mdlName),
                model_name=mdlName)

    ## find optimal positive_prob_treshold
    # https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/

    score_threshold_roc = optimalBalancedThresholdOnROC(fpr, tpr, thresholds_roc)
    score_threshold_pr = optimalBalancedThresholdOnPRCurve(precision, recall, thresholds_pr)

    # thresholds_roc[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
    t_candidates = [x for x in np.linspace(start=max(thresholds_roc[1:].min(), thresholds_pr.min()),
                                           stop=min(thresholds_roc[1:].max(), thresholds_pr.max()),
                                           num=10)] + [score_threshold_roc, score_threshold_pr]

    for i, t in enumerate(t_candidates, 0):

        ypred = adjust_threshold(yscore, t)

        cm, tn, fp, fn, tp = confusionMatrix(ytest, ypred, labels=(0, 1))
        plotConfusionMatrix(cm, classes, normalize=False,
                            result_savepath=result_savepath + '/{}_confusion_matrix_threshold_{}.png'.format(mdlName, i))

        markThresholdOnROC(t, thresholds_roc, fpr, tpr,
                           result_savepath=result_savepath + '/{}_ROC_with_threshold_{}.png'.format(mdlName, i),
                           model_name=mdlName)
        markThresholdOnPRCurve(t, thresholds_pr, precision, recall, no_skill,
                               result_savepath=result_savepath + '/{}_PR_with_threshold_{}.png'.format(mdlName, i),
                               model_name=mdlName)


    ## prediction based on chosen score threshold
    score_threshold = t_candidates[-1]
    ypred = adjust_threshold(yscore, score_threshold)

    ## visualize the prediction
    plotBatchPrediction(ytest, ypred, Xtest, int(img_size/2), classes)

    ## print classification_report
    print(metrics.classification_report(ytest, ypred, labels=[0, 1], target_names=classes, digits=3))

    ## plot confusion matrix
    sns.set(font_scale=1.4)
    cm, tn, fp, fn, tp = confusionMatrix(ytest, ypred, labels=(0, 1))
    plotConfusionMatrix(cm, classes, normalize=False,
                        result_savepath=result_savepath + '/{}_confusion_matrix.png'.format(mdlName))

    ## plot ROC and PR curve
    markThresholdOnROC(score_threshold, thresholds_roc, fpr, tpr,
                       result_savepath=result_savepath + '/{}_ROC_with_threshold.png'.format(mdlName),
                       model_name=mdlName)
    markThresholdOnPRCurve(score_threshold, thresholds_pr, precision, recall, no_skill,
                           result_savepath=result_savepath + '/{}_PR_with_threshold.png'.format(mdlName),
                           model_name=mdlName)


##

