# -*- coding: utf-8 -*-
import time
import numpy as np
from mpmath import mpf, mpc, pi, sin, tan, exp
import matplotlib.pyplot as plt
import functools as ft
import cmath as cm
import sys

startTime = time.time()


# compute the original function f(t) from its Laplace transform F(s)
# sigma is real part of s, needs to be larger than the rightmost pole of F(s)
def invertLPT(F_s, f_max, N=512, sigma=1, plotFT=False):
    '''
    Computes inverse Laplace transform of F(s) using FFT. Parameters are:
    F_s - Laplace Transform to be inverted
    f_max - Nyquist frequency for FFT
    N - Number of sample points on which to evaluate F(s) and f(t)
    sigma - real part of Laplace variable s (signal dampening),
    sigma needs to be larger than the rightmost pole of F(s)
    '''

    # frequencies arranged according to ifft definition of numpy
    ff = np.concatenate((np.linspace(0, f_max, num=N/2),
                        np.linspace(-f_max, 0, num=N/2)))
    dt = 1/(2*f_max)  # time discretization
    # formula for discrete FT from continous FT
    # FT is evaluated at w = sigma + i2pif because we want inverse LPT
    FT = np.array([F_s(sigma + 2j*np.pi*ff[i])/dt for i in range(ff.size)])

    # plotting spectrum of F_(s) in order to choose best f_max
    if plotFT:
        plt.figure(1)
        plt.plot(ff[:N/2], FT[:N/2].real, 'b-')  # positive part
        plt.plot(ff[N/2+1:], FT[N/2+1:].real, 'b-')  # negative part
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude [real part]')

        plt.figure(2)
        plt.plot(ff[:N/2], FT[:N/2].imag, 'g-')  # positive part
        plt.plot(ff[N/2+1:], FT[N/2+1:].imag, 'g-')  # negative part
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude [imaginary part]')
        plt.show()

    '''could also try np.fft.irfft'''
    IFT = np.fft.ifft(FT, n=N)  # inverse FT

    # multiplying by inverse of decaying exponential to obtain original f(t)
    return IFT.real*np.exp(sigma*np.arange(N)*dt)


# test function for numerical inversion of laplace transform
def test_F(s):
    return 1/(1+s**2)

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

def main():

    N = 2**20
    f_max = 8
    sigma = 1
    dt = 1/(2*f_max)
    tt = np.arange(N)*dt
    iLPT = invertLPT(test_F, f_max=f_max, N=N, sigma=sigma, plotFT=False)

    plt.plot(tt[:200], iLPT[:200])
    plt.show()
    sys.exit()


    # # trying numerical inversion through inverse fourier transform
    # # trying for F(s) = 1/s --> f(t) = 1
    # # frequencies arranged according to ifft definition of numpy
    # f_max = 16
    # f = np.concatenate((np.linspace(0, f_max, num=512),
    #                     np.linspace(-f_max, 0, num=512)))
    # N = f.size
    # dt = 1/(2*f_max)
    # # real part of s, needs to be larger than the rightmost pole of F(s)
    # sigma = 1
    # # computing inverse FT of F(sigma + 2j*pi*f) for inverse LT
    # FT = np.array([test_F(sigma + 2j*np.pi*f[i])/dt for i in range(f.size)])
    # IFT = np.fft.ifft(FT)
    #
    # # plt.plot(np.arange(0, FT.size/2), FT[:FT.size/2].real)
    # # plt.plot(np.arange(-FT.size/2+1, 0), FT[FT.size/2+1:].real)
    # # plt.show()
    # # plt.plot(np.arange(0, FT.size/2), FT[:FT.size/2].imag)
    # # plt.plot(np.arange(-FT.size/2+1, 0), FT[FT.size/2+1:].imag)
    # # plt.show()
    # # sys.exit()
    #
    # plt.plot((IFT.real*np.exp(sigma*np.arange(N)*dt))[:32])
    # plt.show()
    # sys.exit()
    #
    # plt.plot(IFT.real)
    # plt.plot(np.exp(-sigma*np.arange(0, N)*dt), 'r--')
    # plt.show()
    # sys.exit()

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
