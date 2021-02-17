import cupy as cp
y_true = [[0, 1], [-1, 1], [7, -6]]
y_pred = [[4, 1], [-1, 5], [2, -6]]
y_true = cp.asarray(y_true)
y_pred = cp.asarray(y_pred)


def check_length(y_true, y_pred):
    if len(y_true) == len(y_pred):
        return True
    else:
        return False


def mean_absolute_error(y_true, y_pred, multioutput='uniform_average'):
    output_errors = cp.average(cp.abs(y_pred - y_true), weights=None, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            multioutput = None

    return cp.average(output_errors, weights=None)


mean_absolute_error(y_true, y_pred)
