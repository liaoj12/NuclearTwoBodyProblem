# -*- coding: utf-8 -*-
"""
Name: Junjie Liao

This module implements a routine that performs Gaussian integration for an
arbitrary function provided by a user-supplied function. It allows for
arbitrary mesh sizes and transformation functions.
"""

import numpy as np


def finite_boundary_gauss_integration(f, a, b, n):
    """
    Function that calculates a definite integral with finite boundaries.

    Parameters
    ----------
    f : function
        a function given by user
    a : double
        lower boundary of the integral
    b : double
        upper boundary of the integral
    n : int
        number of mesh points to be generated

    Returns
    -------
    result : double
             gaussian quadrature of the integral
    """

    # generates mesh points and weights
    x, w = gauleg(a, b, n)

    # sum over x and w, return result
    result = gauss_quadrature(f, x, w)

    return result


def infinite_boundary_gauss_integration(f, n, q0=1.0):
    """
    Function that calculates a definite integral with infinite boundaries.

    Parameters
    ----------
    f : function
        a function given by user
    n : int
        number of mesh points to be generated
    q0 : double
         a parameter that divides the mesh in half, i.e., for an even
         number of mesh points, half the integration points are below
         q0 and half above

    Returns
    -------
    result : double
             gaussian quadrature of the integral
    """

    # generates
    x, w = gauleg(-1, 1, n)

    x_new, w_new = transformation(x, w, q0=q0)

    result = gauss_quadrature(f, x_new, w_new)

    return result


def infinite_boundary_with_singularity_gauss_integration(f, n, singular, q, denom, transformation):
    """
    Function that calculates a definite integral with infinite boundaries.
    Noticed that it uses subtraction method to remove the singularity.

    Parameters
    ----------
    f : function
        a function given by user
    n : int
        number of mesh points to be generated
    singular : double
               value which makes the denom function equal to zero
    q : double
        parameter that divides the mesh in half as possible
    denom : function
            a function appears in the denominator in the integrand
    transformation : function
                     a transformation function to map the infinite boundary
                     to (-1, 1)

    Returns
    -------
    result : double
             gaussian quadrature of the integral

    """

    # generates mesh points and weights
    x, w = gauleg(-1, 1, n)

    # transform x and w with given transformation function
    x_new, w_new = transformation(x, w, q0=q)

    # apply subtraction method, and sum over x and w, return result
    result = gauss_quadrature(lambda x: (f(x)-f(singular))/denom(x, x0=singular),
                              x_new, w_new)

    return result


def gauleg(a, b, n):
    """
    Function that generates mesh points and weights using Gauss-Legendre algorithm.
    Simply rewritten from the Fortran codes given from book "Numerical Recipes".

    Parameters
    ----------
    a : double
        lower boundary of the integral
    b : double
        upper boundary of the integral
    n : int
        number of mesh points to be generated

    Returns
    -------
        x : ndarray
            arrays of mesh points of type double
        w : ndarray
            arrays of weights of type double
    """

    eps = 3.0E-14

    x = np.zeros((n,1), dtype=float)
    w = np.zeros((n,1), dtype=float)

    m = (n+1)/2
    xm = 0.5*(b+a)
    xl = 0.5*(b-a)

    for i in range(0, m):
        z = np.cos(np.pi*(i+1-.25)/(n+.5))
        checked_error = False
        while not checked_error:
            p1 = 1.
            p2 = 0.
            for j in range(1, n+1):
                p3, p2 = p2, p1
                p1 = ((2.*j-1.)*z*p2-(j-1.)*p3)/j
            pp = n*(z*p1-p2)/(z*z-1.)
            z1 = z
            z = z1-p1/pp
            if abs(z-z1) < eps:
                checked_error = True
        x[i] = xm-xl*z
        x[n-1-i] = xm+xl*z
        w[i] = 2.*xl/((1.-z*z)*pp*pp)
        w[n-1-i] = w[i]

    return x, w

def transformation(x, w, q0=1., a=1.):
    """
    Function that transform the integration mesh points and weights to map
    the desire integration boundaries, i.e., (0, infinity) -> (-1, 1).
    Noted that this has to be reasonably smooth strictly monotonic function.

    Parameters:
    x : double
        array of integration mesh points
    w : double
        array of integration weights
    q0 : double
         a parameter that divides the mesh in half, i.e., for an even
         number of mesh points, half the integration points are below
         q0 and half above
    a : double
        a parameter that maps the finite bound, i.e., (1, inf), this case a=1.

    Returns:
    result : double
             transformation of x and w
    """

    result = q0*(a+x)/(1.-x), w*q0*(a+1.)/((1.-x)**2)

    return result

def gauss_quadrature(f, x, w):
    """
    Function that calculates the gaussian quadrature of a given function, with
    the corresponding integration mesh points and weights.

    Args:
        f (function): a function given by user
        x (double, array): integration mesh points
        w (double, array): integration weights
    Returns:
        (double): dot product of w-array and f(x)-array
    """

    return np.dot(w.T, f(x))[0,0]


if __name__ == "__main__":
    # simple definite integral
    print finite_boundary_gauss_integration(lambda x: np.cos(x)**2, 0, 2*np.pi, 6)

    # simple indefinite integral without any singularities
    print infinite_boundary_gauss_integration(lambda x: np.exp(-x**2), 20)

    # principal value integral using subtraction method
    print infinite_boundary_with_singularity_gauss_integration(lambda x: (x**2)*np.exp(-3*(x-2.5)**2),
                                              22, 2., 2, lambda x, x0=2: x**2-x0**2,
                                              transformation)
