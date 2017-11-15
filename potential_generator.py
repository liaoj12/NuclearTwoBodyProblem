# -*- coding: utf-8 -*-
"""
Name: Junjie Liao




This module implements a routine that generates the potential matrix elements
U(p, q) for a given mesh. In addition you need U(p, k) and U(p, beta) for
parameters k and beta that, in general, will be different from any mesh point.
"""

import numpy as np
import gaussian_integration as gi


# useful constant(s)
mV = 41.47  # 41.47 MeV = 1 fm^-2


def mt_potential(r, p_type='I'):
    """
    Function calculate the Malfliet-Tjon type I or type III potentials.

    Note:
    MT model
    I:   VA = 513.968
    II:  VA = 626.885

    Parameters
    ----------
    r : double
        radial distance, in fm
    p_type : string
             type of potential model

    Returns
    -------
    result : double
             the Malfliet-Tjon potential, in 1/fm^2

    """

    # potential strength of attractive part, in MeV fm
    VA = 513.968
    if p_type == 'III':
        VA = 626.885
    # potential strength of repulsive part, in MeV fm
    VR = 1438.720
    # wave number of repulsive part, in 1/fm
    muR = 3.110
    # wave number of attractive part, in 1/fm
    muA = 1.550

    # calculate the Malfliet-Tjon potential
    result = VR*np.exp(-muR*r)/r - VA*np.exp(-muA*r)/r

    return result/mV


def potential_00(p_type='I', mesh_size=48, mesh_parameter=2.0):

    integral = gi.infinite_boundary_gauss_integration(lambda r: r**2*mt_potential(r, p_type=p_type), mesh_size, q0=mesh_parameter)

    u_00 = 2/np.pi * integral

    return u_00


def potential_kk(k, p_type='I', mesh_size=48, mesh_parameter=2.0):

    if k == 0:
        return potential_00(p_type=p_type, mesh_size=mesh_size, mesh_parameter=mesh_parameter)

    integral = gi.infinite_boundary_gauss_integration(lambda r: (np.sin(k*r))**2*mt_potential(r, p_type=p_type), mesh_size,q0=mesh_parameter)

    u_kk = 2/(np.pi*k*k) * integral

    return u_kk


def potential_vector(k_beta=0, p_type='I', mesh_size=48, mesh_parameter=2.0):

    x, w = gi.gauleg(-1, 1, mesh_size)
    x_new, w_new = gi.transformation(x, w, q0=mesh_parameter)

    n, _ = x_new.shape

    result_array = np.zeros((n, 1), dtype=float)
    if k_beta == 0:
        for i, q in enumerate(x_new):
            result_array[i] = gi.infinite_boundary_gauss_integration(lambda r:
                                                                    (2/(np.pi*q))*(r*mt_potential(r, p_type=p_type)*np.sin(q*r)),
                                                                    mesh_size, q0=mesh_parameter)
    else:
        for i, q in enumerate(x_new):
            result_array[i] = gi.infinite_boundary_gauss_integration(lambda r:
                                                                    (2/(np.pi*k_beta*q)*(np.sin(k_beta*r)*mt_potential(r, p_type=p_type)*np.sin(q*r))),
                                                                    mesh_size,
                                                                    q0=mesh_parameter)

    return result_array


def potential_matrix(p_type='I', mesh_size=48, mesh_parameter=2.0):

    x, w = gi.gauleg(-1, 1, mesh_size)
    x_new, w_new = gi.transformation(x, w, q0=mesh_parameter)

    n, _ = x_new.shape

    result_matrix = np.zeros(shape=(n, n), dtype=float)
    for i, q in enumerate(x_new):
        for j, p in enumerate(x_new):
            result_matrix[i, j] = gi.infinite_boundary_gauss_integration(lambda r:
                                                                       (2/(np.pi*p*q)*(np.sin(p*r)*mt_potential(r, p_type=p_type)*np.sin(q*r))),
                                                                       mesh_size,
                                                                       q0=mesh_parameter)

    return result_matrix


if __name__ == "__main__":
    # print "U vec"
    # print potentialVector()

    # print "U mat"
    # singlet = potential_matrix(p_type='I', mesh_size=48, mesh_parameter=2.0)
    # print repr(singlet[0, 0])
    # triplet = potential_matrix(p_type='III', mesh_size=48, mesh_parameter=2.0)
    # print repr(triplet[0, 0])

    potential_kk(0.0)
