import cupy as cp
import warnings
y_true = [2, 1, 0, 0, 1, 0]
y_pred = [2, 1, 2, 0, 0, 1]
y_true = cp.asarray(y_true)
y_pred = cp.asarray(y_pred)
classes = cp.unique(y_true)


def confusion_matrix(actual, predicted, classes):
    matrix = cp.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            matrix[i, j] = cp.sum((actual == classes[i]) &
                                  (predicted == classes[j]))
    return matrix


M = confusion_matrix(y_true, y_pred, classes)


def balanced_accuracy_score(conf_matrix, classes, adjusted=False):
    rec_list = cp.empty(len(classes))
    for i in range(len(classes)):
        TP = conf_matrix[i, i]
        D = 0
        for j in range(0, len(classes)):
            D = D + conf_matrix[i, j]
        try:
            rec = TP/D
            rec_list[i] = rec
        except ZeroDivisionError:
            rec_list[i] = 0
    if cp.any(cp.isnan(rec_list)):
        warnings.warn('y_pred contains classes not in y_true')
        rec_list = rec_list[~cp.isnan(rec_list)]
    balanced_score = cp.sum(rec_list)/len(classes)
    if adjusted:
        n_classes = len(rec_list)
        chance = 1 / n_classes
        balanced_score -= chance
        balanced_score /= 1 - chance
    return balanced_score


balanced_accuracy_score(M, classes)
