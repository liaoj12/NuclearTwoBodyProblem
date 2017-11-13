# -*- coding: utf-8 -*-
"""
Name: Junjie Liao




This module implements a routine that generates the potential matrix elements
U(p, q) for a given mesh. In addition you need U(p, k) and U(p, beta) for
parameters k and beta that, in general, will be different from any mesh point.
"""

import numpy as np
import GaussianIntegration as GI


# useful constant(s)
mV = 41.47  # 41.47 MeV = 1 fm^-2


def MT_Potential(r, model='I'):
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
    model : string
            type of model

    Returns
    -------
    result : double
             the Malfliet-Tjon potential, in 1/fm^2

    """

    # potential strength of attractive part, in MeV fm
    VA = 513.968
    if model == 'III':
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


def potentialVector(k_beta = 0, type='I', mesh_size=48, mesh_parameter=2.0):

    x, w = GI.gauleg(-1, 1, mesh_size)
    x_new, w_new = GI.transformation(x, w, q0=2.0)

    n, _ = x_new.shape

    result_array = np.zeros((n,1), dtype=float)
    if k_beta == 0:
        for i, q in enumerate(x_new):
            result_array[i] = GI.infinite_boundary_gauss_integration(lambda r:
                                                                    (2/(np.pi*q))*(r*MT_Potential(r, model=type)*np.sin(q*r)),
                                                                    mesh_size,
                                                                    q0=mesh_parameter)
    else:
        for i, q in enumerate(x_new):
            result_array[i] = GI.infinite_boundary_gauss_integration(lambda r:
                                                                    (2/(np.pi*k_beta*q)*(np.sin(k_beta*r)*MT_Potential(r, model=type)*np.sin(q*r))),
                                                                    mesh_size,
                                                                    q0=mesh_parameter)

    return result_array


def potentialMatrix(type='I', mesh_size=48, mesh_parameter=2.0):

    x, w = GI.gauleg(-1, 1, mesh_size)
    x_new, w_new = GI.transformation(x, w, q0=mesh_parameter)

    n, _ = x_new.shape

    result_matrix = np.zeros(shape=(n,n), dtype=float)
    for i, q in enumerate(x_new):
        for j, p in enumerate(x_new):
            result_matrix[i,j] = GI.infinite_boundary_gauss_integration(lambda r:
                                                                       (2/(np.pi*p*q)*(np.sin(p*r)*MT_Potential(r, model=type)*np.sin(q*r))),
                                                                       mesh_size,
                                                                       q0=mesh_parameter)

    return result_matrix

if __name__ == "__main__":
    print "U vec"
    print potentialVector()

    # print "U mat"
    # mat = potentialMatrix(type='III')
    # print mat[0:2,:]
