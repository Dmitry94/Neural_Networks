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

def generate_spiral_data(size, classes):
    """
        Generates spiral data in 2D.
        size : int
            Count of points for each class.
        classes : int
            Count of classes.
    """
    np.random.seed(0)

    D = 2
    X = np.zeros((size * classes, D))
    Y = np.zeros(size * classes, dtype='uint8')
    for j in xrange(classes):
        ix = range(size * j, size * (j + 1))
        r = np.linspace(0.0, 1.0, size)
        t = np.linspace(j * 4, (j + 1) * 4, size) + np.random.randn(size) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    return X, Y
