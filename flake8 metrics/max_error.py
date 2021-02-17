#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cupy as cp
import time
y_true = [3, 2, 7, 1]
y_pred = [9, 2, 7, 1]
y_true = cp.asarray(y_true)
y_pred = cp.asarray(y_pred)


def max_error(y_true, y_pred):
    p = time.time()
    error = cp.ElementwiseKernel(
        'T y_pred, T y_true',
        'T diff',
        'diff = y_pred - y_true',
        'error')
    diff = error(y_pred, y_true)
    d = time.time()
    print("Time: ", d-p)
    return cp.max(diff)


max_error(y_true, y_pred)
