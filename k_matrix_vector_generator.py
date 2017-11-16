# -*- coding: utf-8 -*-
"""
Name: Junjie Liao

"""

import potential_generator as pmg
import gaussian_integration as gi


def k_matrix_and_inhomovector(EinWave, p_type='I', mesh_size=48, mesh_parameter=2.0):
    """
    Function generates the K matrix(kernel). If the given energy is positive, it
    will calculate the kernel for scattering; else if it's negative, it will
    calculate the kernel for bound-state.

    Parameters
    ----------
    EinWave : double
              energy in terms of wave number; could be both positive & negative
    p_type : string
             potential type
    mesh_size : int
                adsf
    mesh_parameter : double
                     asdf

    Returns
    -------
    Kmat : ndarray(dtype=float)
           size n x n with double type elements matrix, i.e. q^2(U(p,q)-U(p,betaOrk))/(-a^2ork^2-q^2)
    Uvec : ndarray(dtype=flat)
           size n x 1 with double type elements vector, i.e. U(p,beta) or U(p,k)
    """

    x, w = gi.gauleg(-1, 1, mesh_size)
    x_new, w_new = gi.transformation(x, w, q0=mesh_parameter)
    n, _ = x_new.shape

    u_vec = pmg.potential_vector(k_beta=EinWave, p_type=p_type, mesh_size=mesh_size, mesh_parameter=mesh_parameter)
    k_mat = pmg.potential_matrix(p_type=p_type, mesh_size=mesh_size, mesh_parameter=mesh_parameter)

    for row in range(n):
        k_mat[row, :] -= u_vec[row]
        for col in range(n):
            k_mat[row, col] *= (x_new[col]**2)*w_new[col]
            # E>0 (scattering)
            if EinWave >= 0:
                k_mat[row, col] /= (EinWave**2-x_new[col]**2)
            # E<0 (bound-state)
            elif EinWave < 0:
                k_mat[row, col] /= (-EinWave**2-x_new[col]**2)

    return k_mat, u_vec


if __name__ == "__main__":
    K, _ = k_matrix_and_inhomovector(EinWave=0, p_type='I', mesh_size=48, mesh_parameter=2.0)
    print repr(1-K[0, 0])
