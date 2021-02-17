import cupy as cp
y_true = [0, 1, 1, 0]
y_prob = [0.1, 0.9, 0.8, 0.3]
y_true = cp.asarray(y_true)
y_prob = cp.asarray(y_prob)


def check_length(y_true, y_pred):
    if len(y_true) == len(y_pred):
        return True
    else:
        return False


def brier_score_loss(y_true, y_prob, *, sample_weight=None, pos_label=None):
    if check_length(y_true, y_prob) is True:
        labels = cp.unique(y_true)
        if pos_label is None:
            if (cp.array_equal(labels, [0]) or cp.array_equal(labels, [-1])):
                pos_label = 1
            else:
                pos_label = y_true.maximum()
        y_true = cp.asarray(y_true == pos_label, int)
    return float(cp.average((y_true - y_prob) ** 2, weights=sample_weight))


brier_score_loss(y_true, 1-y_prob, pos_label=0)
