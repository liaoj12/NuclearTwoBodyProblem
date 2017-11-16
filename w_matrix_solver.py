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


def w_solver(g, k):
    """

    Parameters
    ----------
    g : ndarray(dtype=float)
        adsf
    k : ndarray(dtype=float)
        asdf

    Returns
    -------
    f : ndarray(dtype=flat)
        adsf
    """

    n, _ = g.shape

    # direct solution method
    mat = np.eye(n, dtype=float)-k
    augmented_mat = np.append(mat, g, axis=1)
    f, det = gje.Gauss_Jordan_Elimination_with_Pivoting(augmented_mat)

    # check if 1-k is invertible
    if np.fabs(det-0) < 1e-50:
        print "determinant is 0, can't inverse matrix!"
        sys.exit("program abort!")

    return f


def jost(w_vec, k, mesh_size=48, mesh_parameter=2.0):

    x, w = gi.gauleg(-1, 1, mesh_size)
    x_new, w_new = gi.transformation(x, w, q0=mesh_parameter)

    # im_f = np.pi/2 * k * w_kk

    return 0


def scattering_length(p_type='I', mesh_size=48, mesh_parameter=2.0):

    x, w = gi.gauleg(-1, 1, mesh_size)
    _, w_new = gi.transformation(x, w, q0=mesh_parameter)

    # calculate integral of w(q, 0)
    # calculate w(p, 0) first
    k, g = kmvg.k_matrix_and_inhomovector(0, p_type=p_type, mesh_size=mesh_size, mesh_parameter=mesh_parameter)
    w_vec = w_solver(g, k)
    # integrate over it
    integral = w_new.T.dot(w_vec)[0, 0]

    # w(0, 0)
    w_00 = w_vec[0, 0]

    # scattering length
    a = np.pi/2 * w_00 / (1+integral)

    return a


if __name__ == "__main__":
    # K, g = kmvg.k_matrix_and_inhomovector(EinWave=0, p_type='I', mesh_size=48, mesh_parameter=2.0)
    # print w_solver(g, K)

    singlet = scattering_length(mesh_size=47)  # should be around -23.393974
    triplet = scattering_length(p_type='III')  # should be around 5.47181
    print singlet, triplet
