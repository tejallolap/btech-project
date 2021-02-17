import cupy as cp
from cuml.metrics import accuracy_score
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]


def zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None):
    y_true = cp.asarray(y_true)
    y_pred = cp.asarray(y_pred)
    score = accuracy_score(y_true, y_pred, handle=None, convert_dtype=True)
    if normalize:
        return 1 - score
    else:
        if sample_weight is not None:
            samples = cp.sum(sample_weight)
        else:
            samples = len(y_true)
            print(samples)
        return samples - score


zero_one_loss(y_true, y_pred)
