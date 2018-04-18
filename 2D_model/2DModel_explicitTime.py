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
import os
startTime = time.time()


def analysis(result, XY=None, cc=None, tt=None, plot=False, per=0.1, dimX=None,
             dimY=None, dt=0.01, savePath=None):
    '''
    Function analyses results from ls-optimization,
    if given comparison plots of results and original concentration profiles
    cc[i, :, :] at time tt[i] will be made, where D and F is averaged over
    top 'per' percent (standart is 0.1 - averaged over top 10%)
    '''

    # ----------------- setting working parameters --------------------- #
    # saving output in current results folder in current directory
    if savePath is None:
        savePath = os.path.join(os.getcwd(), 'results/')
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    I = result.size  # number of different ls-opti runs
    nbr = np.ceil(per*I).astype(int)  # number for top x% of the runs
    if plot:
        M = tt.size  # number of different profiles

    # only valid for symmetric discretizations so far
    N = int(result[0].x.size/2)  # number of bins
    if dimX is None:  # X and Y dimension
        dimX = np.sqrt(N).astype(int)
    if dimY is None:
        dimY = np.sqrt(N).astype(int)
    if XY is None:
        X, Y = np.meshgrid(np.arange(dimY), np.arange(dimX))
    else:
        X, Y = XY[0], XY[1]
    # ----------------- setting working parameters --------------------- #

    # ------------------------ gathering data -------------------------- #
    # cf. definition of cost for scipy algo
    Error = np.array([np.sqrt(result[i].cost*2) for i in range(I)])
    indices = np.argsort(Error)  # for sorting according to error
    # averaged D and F profiles
    D_avg = np.mean([result[indices[i]].x[:N] for i in range(nbr)], axis=0)
    F_avg_pre = np.mean([result[indices[i]].x[N:] for i in range(nbr)], axis=0)
    # reshaping for correct layout
    D_avg, F_avg_pre = D_avg.reshape(dimX, dimY), F_avg_pre.reshape(dimX, dimY)
    F_avg = F_avg_pre - F_avg_pre[0, 0]  # F[0, 0] = 0 as gauge value
    D_std = np.std([result[indices[i]].x[:N] for i in range(nbr)], axis=0)
    F_std = np.std([result[indices[i]].x[N:] for i in range(nbr)], axis=0)
    # reshaping for correct layout
    D_std, F_std = D_std.reshape(dimX, dimY), F_std.reshape(dimX, dimY)

    # gathering best D and F for computation of profiles
    D_best = result[indices[0]].x[:N]
    F_best_pre = result[indices[0]].x[N:]
    # reshaping for correct layout
    D_best = D_best.reshape(dimX, dimY)
    F_best_pre = F_best_pre.reshape(dimX, dimY)
    F_best = F_best_pre - F_best_pre[0]

    # saving gathered data, a bit messy as we're saving multiple 2D arrays
    headers = ['# Array1: X-Mesh [µm]\n', '# Array2: Y-Mesh [µm]\n',
               '# Array3: average diffusivity [µm^2/s]\n',
               '# Array4: stdev diffusivity [+/- µm^2/s]\n',
               '# Array5: average free energy [k_BT]\n',
               '# Array6: stdev of free energy [+/- k_BT]\n']
    with open(savePath+'DF.txt', 'wb') as outfile:
        # writing header
        outfile.write('# Diffusivity and free energy profiles from '
                      'analysis\n'.encode('utf-8'))
        for i, array in enumerate([X, Y, D_avg, D_std, F_avg, F_std]):
            outfile.write(headers[i].encode('utf-8'))  # write array name
            np.savetxt(outfile, array, fmt='%-7.2f')  # save array

    # saving Error of top x% of runs
    np.savetxt(savePath+'minError.txt', Error[indices[:nbr]], delimiter=',',
               header=('Minimal error for top %.2f %% runs.' % (per*100)))
    # ------------------------ gathering data -------------------------- #

    # --------------- computing and plotting profiles  ----------------- #
    if plot:
        # computing profiles for each timepoint
        ccRes = [computeC(tt[i], cc[0, :, :], D_best, F_best, dt=dt)
                 for i in range(M)]

        # saving concentration profiles
        headers = ['# Array1: X-Mesh [µm]\n', '# Array2: Y-Mesh [µm]\n']
        for i in range(M):
            headers.append('# Array%i: c-profile [µM] for t_%i = %i min\n' %
                           (i, i, tt[i]))
        with open(savePath+'cProfilesComputed.txt', 'wb') as outfile:
            # writing header
            outfile.write('# Numerically computed concentration'
                          ' profiles\n'.encode('utf-8'))
            for i, array in enumerate([X, Y] + ccRes):
                outfile.write(headers[i].encode('utf-8'))  # write array name
                np.savetxt(outfile, array, fmt='%-7.2f')  # save array

        # plotting results
        fig = plt.figure()  # D and F in one figure
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, F_avg, cmap=cm.coolwarm, antialiased=True)
        ax.set_title('Free Energy')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('F [k$_B$T]')
        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(X, Y, D_avg, cmap=cm.coolwarm, antialiased=True)
        ax.set_title('Diffusivity')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('D [bins$^2$/timestep]')
        plt.savefig(savePath+'DF.pdf')

        cMax = np.max([ccRes, cc])
        cMin = np.min([ccRes, cc])
        fig = plt.figure()  # concentration profiles in second figures
        row = np.ceil(M/2)  # number of rows for c-plot
        col = np.floor(M/2)  # number of columns for c-plot

        for i in range(M):
            ax = fig.add_subplot(row, col, i+1, projection='3d')
            l1 = ax.plot_wireframe(X, Y, ccRes[i], antialiased=True,
                                   label='Numerical')
            l2 = ax.plot_wireframe(X, Y, cc[i, :, :], linestyles='dashed',
                                   label='Original', antialiased=True)
            ax.set_title('t = %i' % tt[i])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Concentration')
            ax.set_zlim(cMin, cMax)
        ax.legend(handles=[l1, l2])
        plt.savefig(savePath+'profiles.pdf')
    # --------------- computing and plotting profiles  ----------------- #


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
                              max_nfev=None, verbose=verb)
    return result


def main():
    # |- - - - - - > y             layout is in a way that y-axis points to
    # |(0, 0) (0, 1) ...           the right of matrix and x-axis points down,
    # |(1, 0) ...                  origin (0, 0) is in upper left corner,
    # | ...                        like a coordinate system tilted
    # v x                          at a right angle

    dimX = 5
    dimY = 5
    cInit = 50
    # middle = int(dimX/2)
    # meshgrid function needs X, Y in switchted order for correct ouput
    X, Y = np.meshgrid(np.arange(dimY), np.arange(dimX))

    D = np.ones((dimX, dimY))
    F = np.zeros((dimX, dimY))
    for i in range(dimY):
        F[:, i] = -np.arange(dimX)
    c0 = np.ones((dimX, dimY))*cInit

    # compute profiles from given d and f
    tt = np.array([0, 300, 600, 900])
    c1 = computeC(300, c0, D, F, dt=0.0001)
    c2 = computeC(300, c1, D, F, dt=0.0001)
    c3 = computeC(300, c2, D, F, dt=0.0001)
    cInput = np.array([c0, c1, c2, c3])
    print(np.sum(c0), np.sum(c1), np.sum(c2), np.sum(c3))
    # saving concentration profiles
    cWrite = [c0, c1, c2, c3]
    headers = ['# Array1: X-Mesh [µm]\n', '# Array2: Y-Mesh [µm]\n']
    for i in range(len(cWrite)):
        headers.append('# Array%i: c-profile [µM] for t_%i = %i min\n' %
                       (i, i, tt[i]))
    with open('cProfiles.txt', 'wb') as outfile:
        # writing header
        outfile.write('# Original concentration profiles\n'.encode('utf-8'))
        for i, array in enumerate([X, Y] + cWrite):
            outfile.write(headers[i].encode('utf-8'))  # write array name
            np.savetxt(outfile, array, fmt='%-7.2f')  # save array

    # try to refind d and f using ls-optimization
    boundsD = (0, 10)  # bounds for D
    boundsF = (-10, 10)  # bounds for F
    DInit = np.random.rand(dimX, dimY, 5)*5
    FInit = np.ones((dimX, dimY))*(-5)
    result = np.array([optimization(DInit[:, :, i], FInit, boundsD, boundsF,
                                    cInput, tt, dt=0.0001, verb=2)
                       for i in range(12)])
    np.save('result.npy', result)
    result = np.load('/Users/AmanuelWK/Dropbox/PhD/GitHub/FokkerPlanckModeling/2DModel/result.npy')
    analysis(result, XY=[X, Y], cc=cInput, tt=tt, plot=True, per=1, dt=0.00001)


if __name__ == "__main__":
    main()
    print("Execution time was %.2f minutes"
          % ((time.time() - startTime)/60))
