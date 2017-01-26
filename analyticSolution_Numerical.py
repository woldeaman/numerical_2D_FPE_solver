# -*- coding: utf-8 -*-
# Implementing symbolic back transform of laplace solution found analytically
import time
import numpy as np
from mpmath import mpf, mpc, pi, sin, tan, exp
import matplotlib.pyplot as plt
import functools as ft
import cmath as cm
import sys

startTime = time.time()


# defining laplace transform
# from initial computation
# def Laplace_F(s, x):
#     '''
#      Laplace transform obtained from analytical calculations, parameters
#      were chosen accoring to experimental setup
#     '''
#
#     # variables defining the system
#     # values taken from computational results
#     c0 = 4
#     B = 100
#     H = 600
#     D = 130  # corresponds to D in analytical model
#     d = 40  # corresponds to D' in analytical model
#     F = -1
#
#     # variables defined for symplifications
#     x_Tilde = np.complex_([x * cm.sqrt(s)/np.sqrt(D)])
#     B_Tilde = np.complex_([B * cm.sqrt(s)/np.sqrt(D)])
#     H_Tilde = np.complex_([H * cm.sqrt(s)/np.sqrt(D)])
#     delta = np.sqrt(D/d)
#     k = np.exp(-F)  # Free Energy F in [kB*T]
#
#     # contracted variables from solving PDE in laplace space
#     C = 1/(np.exp(-B_Tilde) - np.exp(B_Tilde))
#     Q = -np.exp(delta*B_Tilde - 2*delta*H_Tilde) * (1/delta + 1/k) - np.exp(-delta*B_Tilde) * C/k
#     R = -np.exp(delta*B_Tilde - 2*delta*H_Tilde) * (1/delta + 1/k) + np.exp(-delta*B_Tilde) * (1/delta - 1/k)
#     K = C - 2*Q/R
#     print(K)
#
#     # solutions in laplace space
#     c_Tilde1 = c0/s * (np.exp(x_Tilde) + (np.exp(-x_Tilde) - np.exp(x_Tilde))/(2 - np.exp(B_Tilde)/K))
#     c_Tilde2 = c0/s * ((np.exp(x_Tilde) * (np.exp(-2*x_Tilde) - np.exp(-2*delta*H_Tilde) - Q))/(R*(K - np.exp(-B_Tilde)/2)))
#
#     if x < 0:
#         return ('Error: not defined for negative x')
#     elif x > H:
#         return ('Error: not defined for x > H')
#     elif (x <= B):
#         return c_Tilde1
#     elif (x > B):
#         return c_Tilde2


# test function for numerical inversion of laplace transform
def test_F(s):
    return 1/s


# Laplace transform of simplified problem
def Laplace_simple(s, x):
    c0 = 4  # c0 at left boundary
    xE = 500  # right end of domain
    D = 25  # D Value from Simulation, in µm^2/s
    F = -1  # F Value from simulation, in kB*T

    # rescaled variables
    c0_Tilde = c0*np.exp(F)
    x_Tilde = x/np.sqrt(D)
    xE_Tilde = xE/np.sqrt(D)

    # Laplace transform
    return c0_Tilde*(cm.exp(x_Tilde*cm.sqrt(s))/s +
                     (cm.exp((xE_Tilde-x_Tilde)*cm.sqrt(s))/s -
                      cm.exp((xE_Tilde+x_Tilde)*cm.sqrt(s))/s) /
                     (2*cm.cosh(xE_Tilde*cm.sqrt(s))))


# numerical inversion of laplace transform
class Talbot(object):
    '''
    Class defined for numerical inversion of laplace transform, taken from:
    http://code.activestate.com/recipes/578799-numerical-inversion-of-the-laplace-transform-with-/
    '''

    def __init__(self, F=test_F, shift=1, N=100):
        self.F = F
        # test = Talbot() or test = Talbot(F) initializes with testfunction F

        self.shift = shift
        # Shift contour to the right in case there is a pole on the
        #   positive real axis :
        # Note the contour will not be optimal since it was originally
        # devoloped
        #   for function with singularities on the negative real axis
        # For example
        #   take F(s) = 1/(s-1), it has a pole at s = 1, the contour
        # needs to be
        #   shifted with one unit, i.e shift  = 1.
        # But in the test example no shifting is necessary

        self.N = N
        # with double precision this constant N seems to best
        # for the testfunction
        #   given. For N = 22 or N = 26 the error is larger (for this special
        #   testfunction).
        # With laplace.py:
        # >>> test.N = 500
        # >>> print test(1) - exp(-1)
        # >>> -2.10032517928e+21
        # Huge (rounding?) error!
        # with mp_laplace.py
        # >>> mp.dps = 100
        # >>> test.N = 500
        # >>> print test(1) - exp(-1)
        # >>> -5e-64

    def __call__(self, t):

        if t == 0:
            print("ERROR:   Inverse transform can not be calculated for t=0")
            return ("Error")

        # Initiate the stepsize
        h = 2*pi/self.N

        ans = 0.0
        # parameters from
        # T. Schmelzer, L.N. Trefethen, SIAM J. Numer. Anal. 45 (2007) 558-571
        c1 = mpf('0.5017')
        c2 = mpf('0.6407')
        c3 = mpf('0.6122')
        c4 = mpc('0', '0.2645')

        # The for loop is evaluating the Laplace inversion at each point
        # theta_i which is based on the trapezoidal rule
        for k in range(self.N):
            theta = -pi + (k+0.5)*h
            z = self.shift + self.N/t*(c1*theta/tan(c2*theta) - c3 + c4*theta)
            dz = self.N/t * (-c1*c2*theta/sin(c2*theta)**2 + c1/tan(c2*theta)+c4)
            ans += exp(z*t)*self.F(z)*dz

        return ((h/(2j*pi))*ans).real


def main():
    # trying numerical inversion through inverse fourier transform
    # trying for F(s) = 1/s --> f(t) = 1
    # frequencies arranged according to ifft definition of numpy
    f = np.concatenate((np.arange(0, 1000), np.arange(-1000, 0)))
    # real part of s, needs to be larger than the rightmost pole of F(s)
    sigma = 1
    # computing inverse FT of F(sigma + 2j*pi*f) for inverse LT
    FT = np.array([test_F(sigma + 2j*np.pi*f[i]) for i in range(f.size)])
    IFT = np.fft.ifft(FT)

    # needed to obtain original function
    F_T = IFT*np.exp(sigma*np.arange(1, IFT.size+1))

    # plt.plot(IFT)
    plt.plot(np.exp(-sigma*np.arange(1, 1000)), 'r--')
    plt.show()

    # # test case of talbot method for inverse laplace transform for F(s)=1/s
    # # --> f(t) = 1, very unstable for larger t
    # test = Talbot(test_F)
    # tt = np.linspace(0, 500, num=100)
    # func = np.array([test(tt[i]) for i in range(tt.size)])
    # print(func)

    # trying numerical inversion of Laplace transform
    # xx = np.linspace(0, 500, num=100)
    # F_s = np.array([ft.partial(Laplace_simple, x=xx[i]) for i in range(xx.size)])
    # inversion = np.array([Talbot(F=F_s[i]) for i in range(xx.size)])
    # tt = np.linspace(1, 600)
    # f = np.array([[inversion[i](tt[j]) for i in range(xx.size)] for j in range(tt.size)])
    # print(f.shape)
    # for i in range(tt.size/10):
    #     plt.plot(xx, f[i*10, :])
    #     plt.show()


if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
