# -*- coding: utf-8 -*-
"""
Name: Junjie Liao


"""

import numpy as np
import sys
import k_matrix_vector_generator as kmvg
import potential_generator as pg
import gauss_jordan_elimination as gje
import gaussian_integration as gi


def w_solver(g, K):
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

    # direct solution method
    mat = np.eye(n, dtype=float)-K
    augmented_mat = np.append(mat, g, axis=1)
    f, det = gje.Gauss_Jordan_Elimination_with_Pivoting(augmented_mat)

    # check if 1-K is invertible
    if np.fabs(det-0) < 1e-50:
        print "determinant is 0, can't inverse matrix!"
        sys.exit("program abort!")

    return f, det


def jost(w_vec, k, mesh_size=48, mesh_parameter=2.0):

    x, w = gi.gauleg(-1, 1, mesh_size)
    x_new, w_new = gi.transformation(x, w, q0=mesh_parameter)

    # im_f = np.pi/2 * k * w_kk

    return 0


def scattering_length(p_type='I', mesh_size=48, mesh_parameter=2.0):

    x, w = gi.gauleg(-1, 1, mesh_size)
    _, w_new = gi.transformation(x, w, q0=mesh_parameter)

    # calculate w(0, 0)
    u_00 = pg.potential_00(p_type=p_type, mesh_size=mesh_size, mesh_parameter=mesh_parameter)
    u_0q = pg.potential_vector(0, p_type=p_type, mesh_size=mesh_size, mesh_parameter=mesh_parameter)
    w_q0 = np.copy(u_0q)

    u_0q[:, 0] -= u_00

    w_00 = (u_00 - u_0q.T.dot(w_q0))[0, 0]

    # calculate integral of w(q, 0)
    integral = np.dot(w_new.T, w_q0)[0, 0]

    a = np.pi/2 * w_00 / (1+integral)

    return a


if __name__ == "__main__":
    # K, g = kmvg.k_matrix_and_inhomovector(EinWave=0, p_type='I', mesh_size=48, mesh_parameter=2.0)
    # print w_solver(g, K)

    print scattering_length(p_type='III')
