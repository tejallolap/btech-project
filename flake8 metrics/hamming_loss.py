import cupy as cp
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
y_true = cp.asarray(y_true)
y_pred = cp.asarray(y_pred)


def check_length(y_true, y_pred):
    if len(y_true) == len(y_pred):
        return True
    else:
        return False


def weight_sum(sample_score, sample_weight, normalize):
    if normalize:
        return cp.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return cp.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()


def hamming_loss(y_true, y_pred, *, sample_weight=None):
    if check_length(y_true, y_pred) is True:
        try:
            return float(weight_sum(y_true != y_pred,
                                    sample_weight, normalize=True))
        except ValueError:
            raise ValueError("{0} is not supported".format(y_true))


hamming_loss(y_true, y_pred)
