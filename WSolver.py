# -*- coding: utf-8 -*-
"""
Name: Junjie Liao


"""

import numpy as np
import KMatVecGenerator as kmvg
import GaussJordanElimination as gje
import GaussianIntegration as gi


def w_solver(g, K, num_of_iter=10):
    """

    Parameters
    ----------
    g : ndarray(dtype=float)
        adsf
    K : ndarray(dtype=float)
        asdf
    num_of_iter : int
                  adsfadsf

    Returns
    -------
    f : ndarray(dtype=flat)
        adsf
    """

    n, _ = g.shape

    # iterative solution (good convergence can be found within 10 steps)
    # f = np.copy(g)
    # for i in range(numOfIter):
    #     print "i: " + str(i)
    #     print repr(f[:3])
    #     f = g + K.dot(f)

    # direct solution
    mat = np.eye(n, dtype=float)-K
    augmented_mat = np.append(mat, g, axis=1)
    f, det = gje.Gauss_Jordan_Elimination_with_Pivoting(augmented_mat)

    return f, det


# def jost(w_vec, k):
#     real_f = 1-int
#     im_f = np.pi/2 * k * w_kk


if __name__ == "__main__":
    K, g = kmvg.KMatrixAndInhomVector(EinWave=0, type='I', mesh_size=48, mesh_parameter=2.0)
    print w_solver(g, K)
