"""
    Some usefull functions/classes for all networks.
"""

import numpy as np

def compute_gradient(F, x):
    """
        Computing gradient F in x using approximate formula.

        Parameters
        -------
            F : functuion from Rn to R
            x : point if Rn

        Returns
            Gradient vector
    """
    grad = np.zeros(x.shape)
    h = 0.00001

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]

        x[ix] = old_value - h
        fxh_m = F(x)

        x[ix] = old_value + h
        fxh_p = F(x)

        x[ix] = old_value
        grad[ix] = (fxh_p - fxh_m) / (2 * h)
        it.iternext()

    return grad
