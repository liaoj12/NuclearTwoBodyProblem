# -*- coding: utf-8 -*-
"""
Name: Junjie Liao


"""

import numpy as np
import KMatVecGenerator as KMVG
import GaussJordanElimination as GJE

def WSolver(g, K, numOfIter=8):
    """

    Parameters
    ----------
    g : ndarray(dtype=float)
        adsf
    K : ndarray(dtype=float)
        asdf

    Returns
    -------
    f : ndarray(dtype=flat)
        adsf
    """

    n, _ = g.shape

    one = np.eye(n, dtype=float)
    trans = np.linalg.inv((one-K))

    f = trans.dot(g)

    return f, det


if __name__ == "__main__":
    K, g = KMVG.KMatrixAndInhomVector(EinWave=1.0)
    WSolver(g, K)
