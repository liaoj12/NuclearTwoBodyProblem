# -*- coding: utf-8 -*-
"""
Name: Junjie Liao

"""

import numpy as np
import PotentialMeshGenerator as PMG
import GaussianIntegration as GI


def KMatrixAndInhomVector(EinWave, mesh_size=48, mesh_parameter=2.0):
    """
    Function generates the K matrix(kernel). If the given energy is positive, it
    will calculate the kernel for scattering; else if it's negative, it will
    calculate the kernel for bound-state.

    Parameters
    ----------
    EinWave : double
              energy in terms of wave number; could be both positive & negative
    mesh_size : int
                adsf
    mesh_parameter : double
                     asdf

    Returns
    -------
    Kmat : ndarray(dtype=float)
           size n x n with double type elements matrix, i.e. q^2(U(p,q)-U(p,betaOrk))/(-a^2ork^2-q^2)
    Uvec : ndarray(dtype=flat)
           size n x 1 with double type elements vector, i.e. U(p,beta)/U(p,k)
    """

    x, w = GI.gauleg(-1, 1, mesh_size)
    x_new, w_new = GI.transformation(x, w, mesh_parameter)
    n, _ = x_new.shape

    Uvec = PMG.potentialVector()
    Kmat = PMG.potentialMatrix()

    for row in range(n):
        for col in range(n):
            Kmat[row,col] = (x_new[row]**2)*(Kmat[row,col]-Uvec[row])*w_new[row]
            # E>0 (scattering)
            if EinWave > 0:
                Kmat[row,col] /= (EinWave**2-x_new[row]**2)
            # E<0 (bound-state)
            elif EinWave < 0:
                Kmat[row,col] /= (-EinWave**2-x_new[row]**2)

    return Kmat, Uvec


if __name__ == "__main__":
    K, g = KMatrixAndInhomVector(EinWave=2.0, mesh_size=48, mesh_parameter=2.0)
    print K[0,:]
    print g.shape
