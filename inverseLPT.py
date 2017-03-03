# -*- coding: utf-8 -*-
import time
import numpy as np
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
        plt.plot(ff[:N//2], FT[:N//2].real, 'b-')  # positive part
        plt.plot(ff[(N//2)+1:], FT[(N//2)+1:].real, 'b-')  # negative part
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude [real part]')

        plt.figure(2)
        plt.plot(ff[:N//2], FT[:N//2].imag, 'g-')  # positive part
        plt.plot(ff[(N//2)+1:], FT[(N//2)+1:].imag, 'g-')  # negative part
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude [imaginary part]')
        plt.show()

    '''could also try np.fft.irfft'''
    IFT = np.fft.ifft(FT, n=N)  # inverse FT

    # multiplying by inverse of decaying exponential to obtain original f(t)
    # # computation in logspac e due to exponential overflow
    # a, b = np.log(IFT.real), sigma*np.arange(N)*dt
    # print(np.min(a))
    # c = a+b
    # return np.exp(c)

    return IFT.real*np.exp(sigma*np.arange(N)*dt)


# test function for numerical inversion of laplace transform
def test_F(s):
    return 4/s


# Laplace transform of simplified problem
def Laplace_simple(s, x):
    c0 = 4  # c0 at left boundary
    xE = 500  # right end of domain
    D = 25  # D Value from Simulation, in µm^2/s
    F = -1  # F Value from simulation, in kB*T

    # rescaled variables
    c0_Tilde = c0*np.exp(-F)
    x_Tilde = x/np.sqrt(D)
    xE_Tilde = xE/np.sqrt(D)

    # Laplace transform
    return c0_Tilde*(cm.cosh((x_Tilde - xE_Tilde)*cm.sqrt(s)) /
                     (s*cm.cosh(xE_Tilde*cm.sqrt(s))))


# Laplace transform of full problem
def Laplace_full(s, x):
    # dimensions in mm because of exp overflow in function and in hours
    # for less memory consumption of inverse laplace transform
    c0 = 4  # c0 at left boundary in [µM]
    # parameter values from simulation for case of positively charged peptide
    xB = 0.10742  # interface location in [mm]
    xE = 0.59082  # right end of domain in [mm], for positively charged peptide
    D_Prime = 128.506*360E-6  # D' Value from Simulation, in mm^2/h
    D = 35.467*360E-6  # D Value from Simulation, in mm^2/h
    F = -0.971  # F Value from simulation, in kB*T

    # # # values for negatively charged peptide
    # xB = 0.10611  # interface location in [mm]
    # xE = 0.61791  # right end of domain in [mm], for negatively charged peptide
    # D_Prime = 572.815*360E-6  # D' Value from Simulation, in mm^2/h
    # D = 107.680*360E-6  # D Value from Simulation, in mm^2/h
    # F = -0.063  # F Value from simulation, in kB*T

    # rescaled variables
    delta = np.sqrt(D_Prime/D)
    k = np.exp(-F)

    # Laplace transform
    if x < xB:
        LPT = ((delta * cm.cosh(cm.sqrt(s/D_Prime)*(x-xB)) +
               k * cm.sinh(cm.sqrt(s/D_Prime)*(x-xB)) *
               cm.tanh(cm.sqrt(s/D)*(xB-xE))) /
               (s*(delta * cm.cosh(cm.sqrt(s/D_Prime)*xB) -
                k * cm.sinh(cm.sqrt(s/D_Prime)*xB) *
                cm.tanh(cm.sqrt(s/D)*(xB-xE)))))

    else:
        LPT = ((cm.cosh(cm.sqrt(s/D)*(x-xE)) /
                cm.cosh(cm.sqrt(s/D)*(xB-xE))) /
               (s*(cm.cosh(cm.sqrt(s/D_Prime)*xB)/k -
                cm.sinh(cm.sqrt(s/D_Prime)*xB) *
                cm.tanh(cm.sqrt(s/D)*(xB-xE))/delta)))

    return c0*LPT


def main():
    # for solution to diffusion problem
    # right end of domain in [mm]
    xE = 0.59082  # for positively charged peptide
    # xE = 0.61791  # for negatively charged peptide
    # places for which to compute concentration
    xx = np.linspace(0, xE, num=100)
    # One Laplace Transform for each x value
    F_s = np.array([ft.partial(Laplace_full, x=xx[i])
                    for i in range(xx.size)])
    # Paramters for inverse FFT
    N = 2**25  # number of frequency samples
    f_max = 300  # Nyquist frequency
    sigma = 1  # real part of laplace variable s, needs to be larger than poles
    dt = 1/(2*f_max)  # time discretization
    tt = np.arange(N)*dt  # time vector in hours (for plotting)
    # for many xx, not storable in one large array, too much memory needed
    # --> export cc[x_i, tt] for each x_i separately and afterwards combine
    np.save('time.npy', tt[:1500])
    for i in range(F_s.size):
        cc = invertLPT(F_s[i], f_max=f_max, N=N, sigma=sigma,
                       plotFT=False)
        np.save('c_x%s.npy' % i, cc[:1500])
        # only save first 1500 entries, corresponding to 150 min of signal
        # no more is needed, as experiments go as far as 15 min

    # combine all exported profiles into one array
    ccProfiles = np.array([np.load('c_x%s.npy' % i) for i in range(F_s.size)])
    np.save('cProfiles.npy', ccProfiles)

    # # # for test function
    # N = 2**24
    # f_max = 80
    # sigma = 1
    # dt = 1/(2*f_max)
    # tt = np.arange(N)*dt
    # iLPT = invertLPT(test_F, f_max=f_max, N=N, sigma=sigma, plotFT=True)
    # plt.plot(tt[:2000], iLPT[:2000], 'b-')
    # # plt.plot(tt[:500], sp.erfc(1/np.sqrt(tt[:500])), 'g-')
    # plt.show()

    # # testing x = 0
    # N = 2**22
    # f_max = 20
    # sigma = 1
    # dt = 1/(2*f_max)
    # tt = np.arange(N)*dt
    # F_s = ft.partial(Laplace_full, x=0)
    # iLPT = invertLPT(F_s, f_max=f_max, N=N, sigma=sigma, plotFT=True)
    # plt.plot(tt[:100], iLPT[:100], 'b-')1
    # plt.show()

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
