# -*- coding: utf-8 -*-
"""
Name: Junjie Liao
Created: Oct 13 2017
Modified Oct 27 2017
Class: Computational Physics III

Execution:
    $ python main.py
"""

import numpy as np
import w_matrix_solver as wms
import k_matrix_vector_generator as kmvg
import matplotlib.pyplot as plt


# energy range between 0-2.4 for k, correspond to 0-100 MeV roughly

def scattering():

    mesh_size = 48
    mesh_parameter = 2.0
    type_1 = 'I'
    type_3 = 'III'
    mv = 41.47

    # calculate scattering lengths for both type-I and type-III MT potential
    a_singlet = wms.scattering_length(type_1, mesh_size, mesh_parameter)
    a_triplet = wms.scattering_length(type_3, mesh_size, mesh_parameter)
    print "The scattering length for type-I model is:"
    print a_singlet
    print "The scattering length for type-III model is:"
    print a_triplet

    # step size
    n = 24
    k_array = np.linspace(0, 2.4, n, dtype=float)
    phase_shift_array_singlet = np.ndarray(shape=(n, 1), dtype=float)
    phase_shift_array_triplet = np.ndarray(shape=(n, 1), dtype=float)

    for i, k in enumerate(k_array):
        kernel_singlet, g_singlet = kmvg.k_matrix_and_inhomovector(k, type_1, mesh_size, mesh_parameter)
        kernel_triplet, g_triplet = kmvg.k_matrix_and_inhomovector(k, type_3, mesh_size, mesh_parameter)

        w_pk_singlet = wms.w_matrix_vector(g_singlet, kernel_singlet)
        w_pk_triplet = wms.w_matrix_vector(g_triplet, kernel_triplet)

        phase_shift_array_singlet[i, 0] = wms.phase_shift(k, w_pk_singlet, type_1, mesh_size, mesh_parameter)
        phase_shift_array_triplet[i, 0] = wms.phase_shift(k, w_pk_triplet, type_3, mesh_size, mesh_parameter)

    print "done and plot"

    # plot
    plt.figure(1)
    plt.plot(k_array*mv, phase_shift_array_singlet)
    plt.savefig('singlet.png')

    plt.figure(2)
    plt.plot(k_array*mv, phase_shift_array_triplet)
    plt.savefig('triplet.png')


def bound():
    pass


def main():
    scattering()
    # bound()


if __name__ == "__main__":
    main()
