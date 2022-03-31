# -*- coding: utf-8 -*-

import numpy as np
import potential_generator as pmg
import gaussian_integration as gi


def k_matrix_and_inhomovector(k, p_type, mesh_size, mesh_parameter):
    """
    Function generates the K matrix(kernel). If the given energy is positive, it
    will calculate the kernel for scattering; else if it's negative, it will
    calculate the kernel for bound-state.

    Parameters
    ----------
    k : double
        energy in terms of wave number; could be both positive or negative
    p_type : string
             potential type
    mesh_size : int
                adsf
    mesh_parameter : double
                     asdf

    Returns
    -------
    kmat : ndarray(dtype=float)
           size n x n with double type elements matrix, i.e. q^2(U(p,q)-U(p,betaOrk))/(-a^2ork^2-q^2)
    u_vec : ndarray(dtype=flat)
            size n x 1 with double type elements vector, i.e. U(p,beta) or U(p,k)
    """

    x, w = gi.gauleg(-1, 1, mesh_size)
    x_new, w_new = gi.transformation(x, w, q0=mesh_parameter)
    n, _ = x_new.shape

    # for positive energy(scattering problem), create U(p,k)
    # for negative energy(bound state problem), create U(p, beta=abs(k))
    if k < 0:
        u_vec = pmg.potential_00(p_type, mesh_size, mesh_parameter)
    else:
        u_vec = pmg.potential_vector(k, p_type, mesh_size, mesh_parameter)

    # generate kernel
    k_mat = np.copy(pmg.potential_matrix(p_type, mesh_size, mesh_parameter))
    for row in range(n):
        k_mat[row, :] -= u_vec[row]
        for col in range(n):
            k_mat[row, col] *= (x_new[col]**2)*w_new[col]
            # E>0 (scattering)
            if k >= 0:
                k_mat[row, col] /= (k**2-x_new[col]**2)
            # E<0 (bound-state) TBD
            elif k < 0:
                k_mat[row, col] /= (-k**2-x_new[col]**2)

    return k_mat, u_vec


if __name__ == "__main__":
    K, _ = k_matrix_and_inhomovector(0, 'I', 48, 2.0)
    print repr(K[0, 1])
