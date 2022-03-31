# -*- coding: utf-8 -*-
import numpy as np
import sys
import k_matrix_vector_generator as kmvg
import potential_generator as pg
import gauss_jordan_elimination as gje
import gaussian_integration as gi


def w_matrix_vector(g, kernel):
    """

    Parameters
    ----------
    g : ndarray(dtype=float)
        inhomogeneous vector
    kernel : ndarray(dtype=float)
             kernel matrix

    Returns
    -------
    f : ndarray(dtype=flat)
        adsf
    """

    n, _ = g.shape

    # direct solution method
    mat = np.eye(n, dtype=float)-kernel
    augmented_mat = np.append(mat, g, axis=1)
    f, det = gje.Gauss_Jordan_Elimination_with_Pivoting(augmented_mat)

    # check if 1-k is invertible
    if det < 1e-16:
        print "determinant is 0, can't inverse matrix!"
        sys.exit("program abort!")

    return f


def w_matrix_kk(k, w_pk, p_type, mesh_size, mesh_parameter):

    x, w = gi.gauleg(-1, 1, mesh_size)
    x_new, w_new = gi.transformation(x, w, q0=mesh_parameter)

    u_pk = pg.potential_vector(k, p_type, mesh_size, mesh_parameter)
    u_kk = pg.potential_kk(k, p_type, mesh_size, mesh_parameter)

    integral = ((x_new**2 * (u_pk - u_kk)) / (k**2 - x_new**2) * w_pk).T.dot(w_new)

    w_kk = u_kk + integral[0, 0]

    return w_kk


def jost_im_scattering(k, w_pk, p_type, mesh_size, mesh_parameter):

    im_f = np.pi/2 * k * w_matrix_kk(k, w_pk, p_type, mesh_size, mesh_parameter)

    return im_f


def jost_re_scattering(k, w_pk, p_type, mesh_size, mesh_parameter):

    x, w = gi.gauleg(-1, 1, mesh_size)
    x_new, w_new = gi.transformation(x, w, q0=mesh_parameter)

    w_kk = w_matrix_kk(k, w_pk, p_type, mesh_size, mesh_parameter)

    integral = (((x_new**2) * w_pk - (k**2) * w_kk)/(k**2 - x_new**2)).T.dot(w_new)

    re_f = 1 - integral[0, 0]

    return re_f


def phase_shift(k, w_pk, p_type, mesh_size, mesh_parameter):

    # Im F(k)
    im_f = jost_im_scattering(k, w_pk, p_type, mesh_size, mesh_parameter)

    # Re F(k)
    re_f = jost_re_scattering(k, w_pk, p_type, mesh_size, mesh_parameter)

    #  calculate
    delta = ((-np.arctan2(im_f, re_f) + 2 * np.pi) % (2 * np.pi)) * 180 / np.pi

    return delta


def scattering_length(p_type, mesh_size, mesh_parameter):

    x, w = gi.gauleg(-1, 1, mesh_size)
    _, w_new = gi.transformation(x, w, q0=mesh_parameter)

    # calculate integral of w(q, 0)
    # generate w(p, 0) first
    kernel, g = kmvg.k_matrix_and_inhomovector(0, p_type, mesh_size, mesh_parameter)
    w_pk = w_matrix_vector(g, kernel)
    # integrate over it with weights
    integral = w_new.T.dot(w_pk)[0, 0]

    # w(0, 0)
    w_00 = w_matrix_kk(0, w_pk, p_type, mesh_size, mesh_parameter)

    # scattering length
    a = np.pi/2 * w_00 / (1+integral)

    return a


def jost_bound(alpha, w_pk, mesh_size, mesh_parameter):

    x, w = gi.gauleg(-1, 1, mesh_size)
    x_new, w_new = gi.transformation(x, w, q0=mesh_parameter)

    integral = ((x_new**2 * w_pk) / (-alpha**2 - x_new**2)).T.dot(w_new)

    f_beta = 1 - integral[0, 0]

    return f_beta


if __name__ == "__main__":
    K, g = kmvg.k_matrix_and_inhomovector(2.3, 'I', 48, 2.0)
    w_pk = w_matrix_vector(g, K)

    # print w_vec[0], w_vec[-1]

    # singlet = scattering_length()
    # triplet = scattering_length(p_type='III')
    # print repr(singlet), repr(triplet)

    print phase_shift(2.3, w_pk, 'I', 48, 2.0)
