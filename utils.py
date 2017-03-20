"""
    Some usefull functions/classes for all networks.
"""

import math
import numpy as np

def calc_l2(point_f, point_s):
    """
        Calculates L2-metriks between 2 points in Rn.
    """
    return math.sqrt(np.sum((point_f-point_s) ** 2))

def calc_l1(point_f, point_s):
    """
        Calculates L1-metriks between 2 points in Rn.
    """
    return np.sum(np.abs(point_f-point_s))
