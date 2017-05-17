# -*- coding: utf-8 -*-
# first start for fokker-planck equation in 2D
import numpy as np
import time
import sys
import scipy.optimize as op
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import functools as ft
from matplotlib import cm
startTime = time.time()


def wRates(xi, yi, xj, yj, D, F, dx):
    '''
    Function computes rates to go from (xi, yi) to (xj, yj) based
    on 2D D and F profiles
    '''
    # using auxillary functions gFunc and fFunc as proposed by
    # Grima et. al - "Accurate discretization of advection-diffusion equations"
    # (2004) PRE https://doi.org/10.1103/PhysRevE.70.036703
    def gFunc(x, y):
        return np.sqrt(D[x, y])*np.exp(F[x, y]/2)  # F in kB*T

    def fFunc(x, y):
        return np.sqrt(D[x, y])*np.exp(-F[x, y]/2)  # F in kB*T

    # valid for n-dimensions
    return (fFunc(xj, yj)*gFunc(xi, yi))/(dx**2)


def computeC(n, c, D, F, dt=0.01, dx=1):
    '''
    Function computes concentration profile after n timesteps
    given an initial concentration profile c0,
    based on F and D profiles with temporal discretization dt and spatial
    discretization dx
    '''

    xMax = c[:, 0].size  # x-dimension
    yMax = c[0, :].size  # y-dimension
    cTot = np.sum(c)  # store total concentration for checks

    # iterative computation of concentration profile
    for t in range(n):
        J = np.zeros((xMax, yMax))
        # J is flux matrix computed from transition-rates
        for x in range(xMax):
            for y in range(yMax):
                # flux coming to bin x, y
                coming = [[wRates(xI, yI, x, y, D, F, dx)*c[xI, yI]
                           if ((xI, yI) != (x, y)
                               and 0 <= xI < xMax
                               and 0 <= yI < yMax) else 0  # sets BCs
                           for xI in range(x-1, x+2)]
                          for yI in range(y-1, y+2)]
                # flux leaving bin x, y
                going = [[wRates(x, y, xI, yI, D, F, dx)*c[x, y]
                          if ((xI, yI) != (x, y)
                              and 0 <= xI < xMax
                              and 0 <= yI < yMax) else 0  # sets BCs
                          for xI in range(x-1, x+2)]
                         for yI in range(y-1, y+2)]
                J[x, y] = (sum(sum(i) for i in coming) -
                           sum(sum(i) for i in going))  # local flux at x, y
        c = c + J*dt  # explicit euler used for temporal discretization
        # check for conserved concentration
        if abs(np.sum(c)-cTot) > 1E-10*np.mean(c):
            print('Careful, concentration is not conserved at timestep %f!'
                  % t)
    return c


def resFun(df, c, tt, dx=1, dt=0.01):
    '''
    Function computes vector of residuals for diffusivity and free energy
    vector df, concentration profiles c[i, :, :] at times tt[i],
    discretization dx and timestep dt.
    '''

    dimX = c[0, :, 0].size  # x-dimension
    dimY = c[0, 0, :].size  # y-dimension
    M = tt.size  # number of profiles at different times
    # reshaping input vector into d and f matrices
    d = np.reshape(df[:int(df.size/2)], (dimX, dimY))
    f = np.reshape(df[int(df.size/2):], (dimX, dimY))

    # computing profiles
    # tt - time, for now still in n timesteps dt
    cComp = [computeC(tt[i], c[0, :, :], d, f, dt, dx) for i in range(M)]
    # normalized residuals
    res = np.zeros((M, dimX, dimY))
    for i in range(M):
        res[i, :, :] = (c[i, :, :] - cComp[i])/np.sqrt(c.size*M)

    return res.reshape(dimX*dimY*M)


def optimization(DInit, FInit, bndsD, bndsF, c, tt, dx=1, dt=0.01, verb=0):
    '''
    Helper function for least squares optimization,
    DInit and FInit are start values for D and F, bdns defines bounds for
    ls-optimization, verb sets verbosity: verb = 0 means no output,
    verb = 1 - output after completed, verb = 2 - output at every iteration
    function returns result object of ls-optimization algorithm
    '''

    optimize = ft.partial(resFun, c=c, tt=tt, dx=dx, dt=dt)
    bounds = (np.concatenate((np.ones(DInit.size)*bndsD[0],
                              np.ones(FInit.size)*bndsF[0])),
              np.concatenate((np.ones(DInit.size)*bndsD[1],
                              np.ones(FInit.size)*bndsF[1])))

    initVal = np.concatenate((DInit.reshape(DInit.size),
                              FInit.reshape(FInit.size)))
    # running freely with standart termination conditions
    result = op.least_squares(optimize, initVal, bounds=bounds,
                              max_nfev=50, verbose=verb)
    return result


def main():
    # |- - - - - - > y             layout is in a way that y-axis points to
    # |(0, 0) (0, 1) ...           the right of matrix and x-axis points down,
    # |(1, 0) ...                  origin (0, 0) is in upper left corner,
    # | ...                        like a coordinate system tilted
    # v x                          at a right angle

    dimX = 10
    dimY = 10
    cInit = 50
    # middle = int(dimX/2)
    # meshgrid function needs X, Y in switchted order for correct ouput
    X, Y = np.meshgrid(np.arange(dimY), np.arange(dimX))

    # D = np.ones((dimX, dimY))
    # F = np.zeros((dimX, dimY))
    F = np.sin(2*X)
    D = abs(np.cos(2*Y))
    c0 = np.ones((dimX, dimY))*cInit

    # compute profiles from given d and f
    c1 = computeC(10, c0, D, F, dt=0.00001)
    c2 = computeC(40, c1, D, F, dt=0.00001)
    c3 = computeC(100, c2, D, F, dt=0.00001)
    cInput = np.array([c0, c1, c2, c3])
    print(np.sum(c0), np.sum(c1), np.sum(c2), np.sum(c3))

    # try to refind d and f using ls-optimization
    boundsD = (0, 10)  # bounds for D
    boundsF = (-10, 10)  # bounds for F
    DInit = np.ones((dimX, dimY))*5
    FInit = np.ones((dimX, dimY))*(-5)
    tt = np.array([0, 10, 50, 150])
    result = optimization(DInit, FInit, boundsD, boundsF, cInput, tt,
                          dt=0.000001, verb=2)
    np.save(result, '/User/AmanuelWK/Desktop/result.npy')

    # plotting results
    # cMax = np.max([c0, c1, c2, c3])
    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # ax.plot_surface(X, Y, F, cmap=cm.coolwarm, antialiased=True)
    # ax.set_title('Free Energy')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('F [kBT]')
    # ax = fig.add_subplot(122, projection='3d')
    # ax.plot_surface(X, Y, D, cmap=cm.coolwarm, antialiased=True)
    # ax.set_title('Diffusivity')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('D [bins$^2$/timestep]')
    # plt.savefig('landscapes.pdf')
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(221, projection='3d')
    # ax.plot_surface(X, Y, c0, vmin=0, vmax=np.max(c0), cmap=cm.coolwarm,
    #                 antialiased=True)
    # ax.set_title('t = 0')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Concentration')
    # ax.set_zlim(0, cMax)
    # ax2 = fig.add_subplot(222, projection='3d')
    # ax2.plot_surface(X, Y, c1, vmin=0, vmax=np.max(c1), cmap=cm.coolwarm,
    #                  antialiased=True)
    # ax2.set_title('t = 10')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Concentration')
    # ax2.set_zlim(0, cMax)
    # ax2 = fig.add_subplot(223, projection='3d')
    # ax2.plot_surface(X, Y, c2, vmin=0, vmax=np.max(c2), cmap=cm.coolwarm,
    #                  antialiased=True)
    # ax2.set_title('t = 50')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Concentration')
    # ax2.set_zlim(0, cMax)
    # ax2 = fig.add_subplot(224, projection='3d')
    # ax2.plot_surface(X, Y, c3, vmin=0, vmax=np.max(c3), cmap=cm.coolwarm,
    #                  antialiased=True)
    # ax2.set_title('t = 100')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Concentration')
    # ax2.set_zlim(0, cMax)
    # # Add a color bar which maps values to colors.
    # # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig('profiles.pdf')
    # plt.show()


if __name__ == "__main__":
    main()
    print("Execution time was %.2f minutes"
          % ((time.time() - startTime)/60))
