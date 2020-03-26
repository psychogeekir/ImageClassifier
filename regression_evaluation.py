import numpy as np
from sklearn import metrics


def regression_metrics(y_true, y_pred, metric_names, name_funcs_dict=None):
    # use sklearn.metrics to compute metrics
    # y_pred and y_true are nd arrays
    # the larger the better

    if name_funcs_dict is None:
        name_funcs_dict = {
            'explained_variance': lambda y_true, y_pred: metrics.explained_variance_score(y_true, y_pred),
            'neg_max_error': lambda y_true, y_pred: - metrics.max_error(y_true, y_pred),
            'neg_mean_absolute_error': lambda y_true, y_pred: - metrics.mean_absolute_error(y_true, y_pred),
            'neg_mean_squared_error': lambda y_true, y_pred: - metrics.mean_squared_error(y_true, y_pred, squared=True),
            'neg_root_mean_squared_error': lambda y_true, y_pred: - metrics.mean_squared_error(y_true, y_pred, squared=False),
            'neg_mean_squared_log_error': lambda y_true, y_pred: - metrics.mean_squared_log_error(y_true, y_pred),
            'neg_median_absolute_error': lambda y_true, y_score: - metrics.median_absolute_error(y_true, y_pred),
            'r2': lambda y_true, y_pred: metrics.r2_score(y_true, y_pred),
            'neg_mean_poisson_deviance': lambda y_true, y_pred: - metrics.mean_poisson_deviance(y_true, y_pred),
            'neg_mean_gamma_deviance': lambda y_true, y_pred: - metrics.mean_gamma_deviance(y_true, y_pred),
        }

    scores = {}
    for metric_name in metric_names:
        scores[metric_name] = name_funcs_dict[metric_name](y_true, y_pred)

    return scores

