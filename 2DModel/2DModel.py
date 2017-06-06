# -*- coding: utf-8 -*-
# first start for fokker-planck equation in 2D
import numpy as np
import time
import sys
import scipy.optimize as op
import matplotlib.pyplot as plt
import scipy.linalg as al
import numpy.linalg as la
import scipy.special as sp
from mpl_toolkits.mplot3d.axes3d import Axes3D
import functools as ft
from matplotlib import cm
import os
startTime = time.time()


def plotting(X, Y, F, D, tt, cc, ccRes=None, savePath=''):
    # plotting results
    fig = plt.figure()  # D and F in one figure
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, F, cmap=cm.coolwarm, antialiased=True)
    ax.set_title('Free Energy')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F [k$_B$T]')
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(X, Y, D, cmap=cm.coolwarm, antialiased=True)
    ax.set_title('Diffusivity')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('D [bins$^2$/timestep]')
    plt.savefig(savePath+'DF.pdf')

    if ccRes is not None:
        cMax = np.max([ccRes, cc])
        cMin = np.min([ccRes, cc])
    else:
        cMax = np.max(cc)
        cMin = np.min(cc)

    M = tt.size
    fig = plt.figure()  # concentration profiles in second figures
    row = np.ceil(M/2)  # number of rows for c-plot
    col = np.floor(M/2)  # number of columns for c-plot

    for i in range(M):
        ax = fig.add_subplot(row, col, i+1, projection='3d')
        if ccRes is not None:
            l1 = ax.plot_wireframe(X, Y, ccRes[i], antialiased=True,
                                   label='Numerical')
        else:
            l1 = ''
        l2 = ax.plot_wireframe(X, Y, cc[i, :, :], linestyles='dashed',
                               label='Original', antialiased=True)
        ax.set_title('t = %i' % tt[i])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Concentration')
        ax.set_zlim(cMin, cMax)
    ax.legend(handles=[l1, l2])
    plt.savefig(savePath+'profiles.pdf')


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
        ccRes = [computeC(tt[i], cc[0, :, :].reshape(dimX*dimY),
                          D_best, F_best, dt=dt).reshape(dimX, dimY)
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
        # plotting D, F and computed and original profiles
        plotting(X, Y, F_avg, D_avg, cc, ccRes, tt)
    # --------------- computing and plotting profiles  ----------------- #


def WMatrix(D, F, dimY=None, dx=1):
    '''
    Function computes rate matrix for 2D systems, here the reduced index i
    is used defined as i = dimY*x + y, with dimY - number of bins in y-disc.
    D and F have to be reduced to 1D-index too in order for this to work!
    '''
    # using auxillary functions gFunc and fFunc as proposed by
    # Grima et. al - "Accurate discretization of advection-diffusion equations"
    # (2004) PRE https://doi.org/10.1103/PhysRevE.70.036703
    def g(i):
        return np.sqrt(D[i])*np.exp(F[i]/2)  # F in kB*T

    def f(i):
        return np.sqrt(D[i])*np.exp(-F[i]/2)  # F in kB*T
    # wRate(i --> j) = (f(j)*g(i))/(dx**2)

    N = D.size  # size of WMatrix is (dimX*dimY)x(dimX*dimY)
    if dimY is None:
        dimY = np.sqrt(N)  # equal discretization assumed if not specified

    #  first compute transition rates to come to bin i
    W = np.array([[f(i-1)*g(j-1)/(dx**2)
                   # only possible to go to neighbouring bins
                   if(
                       # nearest neighbours for first column
                       ((i-1) % dimY == 0 and
                        (abs(j-i) == dimY or
                         j == i+1 or
                         j == i+dimY+1 or
                         j == i-dimY+1))
                       or
                       # nearest neighbours for last column
                       (i % dimY == 0 and
                        (abs(j-i) == dimY or
                         j == i-1 or
                         j == i+dimY-1 or
                         j == i-dimY-1))
                       or
                       # nearest neighbours central bins
                       (((i-1) % dimY != 0 and i % dimY != 0) and
                        (abs(j-i) == dimY or
                         abs(j-i) == 1 or
                         abs(j-i-dimY) == 1 or
                         abs(j-i+dimY) == 1))
                       )
                   else 0
                   for j in range(1, N+1)] for i in range(1, N+1)])
    # indices are shifted because of modulo operator, doesn't handle 0 properly

    # then add rates to leave on main diagonal from original matrix
    for i in range(N):
        W[i, i] = -np.sum([W[j, i] for j in range(N)])

    if np.any(np.sum(W, 0) > 1E-10):
        print('WMatrix not row stochastic!')

    return W


def computeC(cc, tt, W=None, T=None, D=None, F=None, dimY=None, dx=1):
    '''
    Calculates concentration profiles at time t from W or T matrix with
    reflective boundaries, based on concentration profile cc
    '''
    # calculate only variables that are not given
    if T is None:
        if W is None:
            if (D is None) or (F is None):
                print('Error: You gotta give me something... no T, W, D or '
                      'F given!')
                sys.exit()
            else:
                W = WMatrix(D, F, dimY=dimY, dx=dx)
                T = al.expm(W)  # exponential of W
        else:
            T = al.expm(W)  # exponential of W

    return np.dot(la.matrix_power(T, tt), cc)


def resFun(df, cc, tt, dx=1):
    '''
    Function computes vector of residuals for diffusivity and free energy
    vector df, concentration profiles c[i, :, :] at times tt[i] and
    discretization dx width
    '''

    dimX = cc[0, :, 0].size  # x-dimension
    dimY = cc[0, 0, :].size  # y-dimension
    M = tt.size  # number of profiles at different times
    # gathering d and f
    d = df[:int(df.size/2)]
    f = df[int(df.size/2):]
    # reshaping concentration matrix to only work with vectors
    cc = [cc[i, :, :].reshape(dimX*dimY) for i in range(M)]

    # computing profiles
    W = WMatrix(d, f, dimY=dimY, dx=dx)
    T = al.expm(W)
    # normalized residuals
    n = int(sp.binom(M, 2))  # number of combinations for different c-profiles
    res = np.zeros((n, dimX*dimY))
    k = 0
    for i in range(M):
        for j in range(M):
            if i > j:
                res[k, :] = ((cc[i] - computeC(cc[j], tt[i]-tt[j], T=T))
                             / np.sqrt(cc[i].size*n))
                k = k+1

    return res.reshape(res.size)


def optimization(DInit, FInit, bndsD, bndsF, c, tt, dx=1, verb=0):
    '''
    Helper function for least squares optimization,
    DInit and FInit are start values for D and F, bdns defines bounds for
    ls-optimization, verb sets verbosity: verb = 0 means no output,
    verb = 1 - output after completed, verb = 2 - output at every iteration
    function returns result object of ls-optimization algorithm
    '''

    optimize = ft.partial(resFun, c=c, tt=tt, dx=dx)
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

    # defining grid and initial distribution
    dimX = 5
    dimY = 5
    cInit = 4
    c0 = np.zeros((dimX, dimY))
    c0[0, 0] = cInit
    # meshgrid function needs X, Y in switchted order for correct ouput
    X, Y = np.meshgrid(np.arange(dimY), np.arange(dimX))

    # setting D and F
    D = np.ones((dimX, dimY))/100
    # for i in range(dimX):
        # D[:, ]
    F = np.zeros((dimX, dimY))
    # for i in range(dimY):
        # F[:, i] = -np.arange(dimX)
    print(F)

    # compute profiles from given d and f
    tt = np.array([0, 300, 600, 900])

    cInput = [computeC(c0.reshape(c0.size), tt[i], D=D.reshape(D.size), F=F.reshape(F.size),
                       dimY=dimY) for i in range(tt.size)]
    print([np.sum(cInput[i]) for i in range(len(cInput))])

    plotting(X, Y, F, D, tt, np.array(cInput))
    # saving concentration profiles
    # cWrite = [c0, c1, c2, c3]
    # headers = ['# Array1: X-Mesh [µm]\n', '# Array2: Y-Mesh [µm]\n']
    # for i in range(len(cWrite)):
    #     headers.append('# Array%i: c-profile [µM] for t_%i = %i min\n' %
    #                    (i, i, tt[i]))
    # with open('cProfiles.txt', 'wb') as outfile:
    #     # writing header
    #     outfile.write('# Original concentration profiles\n'.encode('utf-8'))
    #     for i, array in enumerate([X, Y] + cWrite):
    #         outfile.write(headers[i].encode('utf-8'))  # write array name
    #         np.savetxt(outfile, array, fmt='%-7.2f')  # save array
    #
    # # try to refind d and f using ls-optimization
    # boundsD = (0, 10)  # bounds for D
    # boundsF = (-10, 10)  # bounds for F
    # DInit = np.random.rand(dimX, dimY, 12)*5
    # FInit = np.ones((dimX, dimY))*(-5)
    # result = np.array([optimization(DInit[:, :, i], FInit, boundsD, boundsF,
    #                                 cInput, tt, dt=0.000001, verb=2)
    #                    for i in range(12)])
    # np.save('result.npy', result)
    # result = np.load('/Users/AmanuelWK/Dropbox/PhD/GitHub/FokkerPlanckModeling/2DModel/result.npy')
    # analysis(result, XY=[X, Y], cc=cInput, tt=tt, plot=True, per=1, dt=0.00001)


if __name__ == "__main__":
    main()
    print("Execution time was %.2f minutes"
          % ((time.time() - startTime)/60))
