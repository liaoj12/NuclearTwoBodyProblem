# -*- coding: utf-8 -*-
"""
This module implements the Gauss-Jordan elimination algorithm with pivoting.
It also includes couple test cases to demonstrate the implmented algorithm.

Note that, this modules only considers the augmented matrix that has exactly one
and only one solution, otherwise, it will quit the program.
"""

import numpy as np
import sys

def pivoting(A, k, N):
    """
    Function that finds the pivot based on the largest magnitude of the element.

    Parameters
    ----------
    A : ndarray
        N x (N+1) size augmented matrix with double type elements.
    k : int
        value of a row number
    N : int
        number of rows of A

    Returns
    -------
    pivot : int
            value of pivot number
    """

    # find the largest a_j1 in magnitude
    pivot = k
    largest = 0

    for i in range(k, N):
        if abs(A[i,k]) > largest:
            largest = abs(A[i,k])
            pivot = i

    return pivot

def swap_rows(A, i, j):
    """
    Function that swap rows i, j for matrix A.

    Parameters
    ----------
    A : ndarray
        N x (N+1) size augmented matrix with double type elements.
    i : int
        row number i
    j : int
        row number j
    """

    # rows swap
    A[j, :], A[i, :] = A[i, :], A[j, :].copy()

def elimination(A, k, N):
    """
    Function that performs the gaussian elimination algorithm.

    Parameters
    ----------
    A : ndarray
        N x (N+1) size augmented matrix with double type elements.
    k : int
        value of pivot row
    N : int
        number of row of matrix A
    """

    # loop through rows
    for i in range(k+1, N):
        coefficient = A[i, k]/A[k, k]
        # loop through columns
        for j in range(k, N+1):
            A[i, j] = A[i, j] - coefficient*A[k, j]

def back_sub_solution(A, N):
    """
    Function that performs back-substitution algorithm.

    Parameters
    ----------
    A : ndarray
        N x (N+1) size augmented matrix with double type elements.
    N : int
        number of row of matrix A

    Returns
    -------
    x : ndarray
        a solution of one-dimension array with double type elements
    """

    # back substitution to find solution
    x = np.ndarray(shape=(N,1), dtype=float)

    # check if solution exists
    if A[N-1,N-1] == 0:
        if A[N-1,N] == 0:
            sys.exit("Infinite many solutions. Quit.")
        else:
            sys.exit("No solution exists. Quit.")


    x[N-1,0] = A[N-1,N]/A[N-1,N-1]

    for k in range(N-2, -1, -1):
        tmp = 0
        for i in range(k+1, N):
            tmp += A[k,i]*x[i,0]
        x[k,0] = 1/A[k,k]*(A[k,N]-tmp)
        # if np.isnan(x[k,0]):
        #     print "A"
        #     print A
        #     print
        #     print "k"
        #     print k
        #     print "x"
        #     print x
        #     sys.exit("quit")

    return x


def Gauss_Jordan_Elimination_with_Pivoting(A):
    """
    Function that performs Gauss-Jordan elimination with pivoting algorithm.

    Parameters
    ----------
    A : np.ndarray
        N x (N+1) size augmented matrix with double type elements.

    Returns
    -------
    x : ndarray
        solution of matrix A
    det : double
        determinant of matrix A
    """

    N, N1 = A.shape  #  size of matrix A

    # check if the dimension of A is N x (N+1)
    if N1-N != 1:
        sys.exit("The dimension of matrix given is wrong. Quit.")

    S = 0  # total number of row swap

    # reduce A into upper triangular form
    for i in range(N):
        # 1. find the pivot row
        pivot = pivoting(A, i, N)

        # 2. swap rows pivot and 0
        # check if one requires to swap
        if pivot != i:
            S += 1  # increment number of rows swap by 1
        swap_rows(A, pivot, i)

        # 3. elimination
        elimination(A, i, N)

    # calculate the determinant of it
    det = (-1)**S
    for i in range(N):
        det *= A[i,i]

    # calculate the solution vector x
    x = back_sub_solution(A, N)

    return x, det


if __name__ == "__main__":

    # test case that has one solution
    A = np.ndarray(shape=(3, 4), dtype=float)
    A[0,0] = 1
    A[0,1] = 1
    A[0,2] = 1
    A[0,3] = 5
    A[1,0] = 2
    A[1,1] = 3
    A[1,2] = 5
    A[1,3] = 8
    A[2,0] = 4
    A[2,1] = 0
    A[2,2] = 5
    A[2,3] = 2

    print Gauss_Jordan_Elimination_with_Pivoting(A)

    # test case that has one solution
    B = np.ndarray(shape=(4, 5), dtype=float)
    B[0,0] = 1
    B[0,1] = 1
    B[0,2] = 2
    B[0,3] = 0
    B[0,4] = 1
    B[1,0] = 2
    B[1,1] = -1
    B[1,2] = 0
    B[1,3] = 1
    B[1,4] = -2
    B[2,0] = 1
    B[2,1] = -1
    B[2,2] = -1
    B[2,3] = -2
    B[2,4] = 4
    B[3,0] = 2
    B[3,1] = -1
    B[3,2] = 2
    B[3,3] = -1
    B[3,4] = 0

    print Gauss_Jordan_Elimination_with_Pivoting(B)

    # test case that has no solution
    # C = np.ndarray(shape=(3, 4), dtype=float)
    # C[0,0] = 1
    # C[0,1] = 2
    # C[0,2] = -3
    # C[0,3] = 2
    # C[1,0] = 6
    # C[1,1] = 3
    # C[1,2] = -9
    # C[1,3] = 6
    # C[2,0] = 7
    # C[2,1] = 14
    # C[2,2] = -21
    # C[2,3] = 13
    #
    # Gauss_Jordan_Elimination_with_Pivoting(C)

    # test case that has infinite solutions
    # D = np.ndarray(shape=(3, 4), dtype=float)
    # D[0,0] = 0
    # D[0,1] = 4
    # D[0,2] = 1
    # D[0,3] = 2
    # D[1,0] = 2
    # D[1,1] = 6
    # D[1,2] = -2
    # D[1,3] = 3
    # D[2,0] = 4
    # D[2,1] = 8
    # D[2,2] = -5
    # D[2,3] = 4
    # Gauss_Jordan_Elimination_with_Pivoting(D)
