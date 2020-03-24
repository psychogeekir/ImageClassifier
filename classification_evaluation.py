import numpy as np
from sklearn import metrics


def confusionMatrix(ytest, ypred, labels=(0, 1)):
    cm = metrics.confusion_matrix(ytest, ypred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    print('number of True Negative: {}'.format(tn))
    print('number of False Positive: {}'.format(fp))
    print('number of False Negative: {}'.format(fn))
    print('number of True Positive: {}'.format(tp))
    return cm, tn, fp, fn, tp


def adjust_threshold(yscore, t):
    # adjust class prediction based on the decision threshold
    # from score to label
    # for binary classification
    return [1 if y >= t else 0 for y in yscore]


def optimalBalancedThresholdOnROC(fpr, tpr, thresholds):
    # locate the threshold with the optimal balance between false positive and true positive rates
    # using Youden’s J statistic.
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print('Best Threshold to Balance FPR and TPR on ROC = {:.3f}, FPR = {:.3f}, TPR = {:.3f}'.format(best_thresh, fpr[ix], tpr[ix]))
    return best_thresh


def optimalBalancedThresholdOnPRCurve(precision, recall, thresholds):
    # locate the threshold with the optimal balance between precision and recall
    # using F1 measure

    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    best_thresh = thresholds[ix]
    print('Best Threshold to Balance Precision and Recall = {:.3f}, F1 score = {:.3f}'.format(best_thresh, fscore[ix]))
    return best_thresh


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

            # ROC curve: binary only
            'roc_curve': lambda y_true, y_score: metrics.roc_curve(y_true, y_score, pos_label=1),
            # Precision-Recall curve
            'pr_curve': lambda y_true, y_score: metrics.precision_recall_curve(y_true, y_score, pos_label=1),

            'pr_auc': lambda y_true, y_score: calPRAUC(y_true, y_score, pos_label=1),
        }

    scores = {}
    for metric_name in metric_names:
        if metric_name == 'neg_brier_score' or metric_name == 'neg_log_loss':
            scores[metric_name] = name_funcs_dict[metric_name](y_true, y_prob)
        elif 'roc_auc' in metric_name or 'average_precision' in metric_name or 'pr_auc' in metric_name:
            scores[metric_name] = name_funcs_dict[metric_name](y_true, y_score)
        elif metric_name == 'roc_curve' or metric_name == 'pr_curve':
            # roc_curve: fpr, tpr
            # pr_curve: precision, recall
            z1, z2, thresholds = name_funcs_dict[metric_name](y_true, y_score)
            scores[metric_name] = (z1, z2, thresholds)
        else:
            scores[metric_name] = name_funcs_dict[metric_name](y_true, y_pred)

    return scores


def calPRAUC(y_true, y_score, pos_label=1):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label=pos_label)
    return metrics.auc(recall, precision)