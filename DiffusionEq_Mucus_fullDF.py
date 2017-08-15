# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# # use this for matplotlib on the cluster
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import time
import inputOutput as io
import FPModel as fp
import scipy.linalg as al
import numpy.linalg as la
import scipy.special as sp
import functools as ft
import scipy.optimize as op
import plottingScripts as ps
import os
import sys

startTime = time.time()


def analysis(result, c0, xx=None, cc=None, tt=None, plot=False, per=0.1,
             savePath=None):
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

    N = cc[:, 0].size  # number of bins
    n = int(sp.binom(M, 2))  # number of combinations for different c-profiles
    if xx is None:
        xx = np.arange(N)
    deltaX = abs(xx[0] - xx[1])
    # ----------------- setting working parameters --------------------- #

    # -------------------------- loading results --------------------------- #
    # gathering data from simulations

    # loading error values, factor two, because of cost function definition
    Error = np.array([np.sqrt(2*result[i].cost / (n*N))
                      for i in range(I)])
    indices = np.argsort(Error)  # for sorting according to error

    # gathering mean of F and D for best 1% of runs
    D_pre = np.mean(np.array([result[indices[i]].x[:N+1]
                              for i in range(nbr)]), axis=0)
    # average over D because only D in between bins is of importance
    D = np.array([(D_pre[i] + D_pre[i+1])/2 for i in range(D_pre.size-1)])
    # TODO: fix this stupid workaround for equal length in D and F
    D = np.append(D, D[-1])
    F_pre = np.mean(np.array([result[indices[i]].x[N+1:]
                              for i in range(nbr)]), axis=0)
    F = F_pre - F_pre[0]
    # gathering standart deviation for top 1% of runs
    DSTD_pre = np.std(np.array([result[indices[i]].x[:N+1]
                                for i in range(nbr)]), axis=0)
    # average over D because only D in between bins is of importance
    DSTD = np.array([(DSTD_pre[i] + DSTD_pre[i+1])/2
                     for i in range(DSTD_pre.size-1)])
    # TODO: fix this stupid workaround for equal length in D and F
    DSTD = np.append(DSTD, DSTD[-1])
    FSTD = np.std(np.array([result[indices[i]].x[N+1:]
                            for i in range(nbr)]), axis=0)

    # gathering best D and F for computation of profiles
    D_best_pre = result[indices[0]].x[:N+1]
    D_best = np.array([(D_best_pre[i] + D_best_pre[i+1])/2  # average D
                       for i in range(D_best_pre.size-1)])
    # TODO: fix this stupid workaround for equal length in D and F
    D_best = np.append(D_best, D_best[-1])
    F_best_pre = result[indices[0]].x[N+1:]
    F_best = F_best_pre - F_best_pre[0]

    # computing WMatrix for open boundary conditions
    W, W10 = fp.WMatrix(D_best_pre, F_best_pre, deltaX=deltaX, bc='open1side')
    # using D_pre here because original D is used for WMatrix computation
    # but averaged D is physically important

    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[:, 0], tt[j], W=W, W10=W10, c0=c0,
                               bc='open1side') for j in range(tt.size)]).T
    # -------------------------- loading results --------------------------- #

    # --------------------------- saving data ------------------------------- #
    # extended xx and cc vector for boundary condition
    xx_ext = np.concatenate((-deltaX*np.ones(1), xx))
    ccRes_ext = np.concatenate((np.ones((1, 4))*c0, ccRes))

    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'concentrationRes.txt', np.c_[xx_ext, ccRes_ext],
               delimiter=',',
               header=('Numerically computed concentration profiles\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: c-profile [micro_M] for t_0 = %i min\n'
                       'cloumn3: c-profile [micro_M] for t_1 = %i min\n'
                       'cloumn4: c-profile [micro_M] for t_2 = %i min\n'
                       'cloumn5: c-profile [micro_M] for t_3 = %i min\n' %
                        (tt[0]/60, tt[1]/60, tt[2]/60, tt[3]/60)))
    # saving averaged DF
    np.savetxt(savePath+'DF.txt', np.c_[xx_ext, D, DSTD, F, FSTD],
               delimiter=',',
               header=('Diffusivity and free energy profiles from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: average diffusivity [micro_m^2/s]\n'
                       'cloumn3: stdev of diffusivity [+/- micro_m^2/s]\n'
                       'cloumn4: average free energy [k_BT]\n'
                       'cloumn5: stdev of free energy [+/- k_BT]\n'))
    # saving best DF
    np.savetxt(savePath+'DF_best.txt', np.c_[xx_ext, D_best, F_best],
               delimiter=',',
               header=('Diffusivity and free energy profiles with lowest '
                       'error from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: diffusivity [micro_m^2/s]\n'
                       'cloumn3: free energy [k_BT]\n'))

    # saving Error of top 1% of runs
    np.savetxt(savePath+'minError.txt', Error[indices[:nbr]], delimiter=',',
               header=('Minimal error for top %.2f %% runs.' % (per*100)))
    # --------------------------- saving data ------------------------------- #

    # ------------------------- plotting data ------------------------------- #
    if plot:
        # plotting profiles
        ps.plotCon(xx, cc, ccRes, tt, c0=c0, save=True, path=savePath)
        # plotting averaged D and F
        ps.plotDF(xx, D, F, D_STD=DSTD, F_STD=FSTD, save=True, path=savePath,
                  scale='log')


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, tt, deltaX=1, c0=None, verb=False):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include: discretization width: deltaX,
    distance of transition regime: dist and concentration at left boundary: c0
    '''

    M = cc[0, :].size  # number of concentration profiles
    N = cc[:, 0].size  # number of bins

    # N paramters to be optimized, D', D and F
    d = df[:(N+1)]
    f = df[(N+1):]  # letting F completely free
    # calculating W and T matrix and extra variables for open BCs
    W, W10 = fp.WMatrix(d, f, deltaX, bc='open1side')
    # catching singular matrix exception
    try:
        Q = la.inv(W)  # inverse of W
    except la.linalg.LinAlgError:
        print('Values for which singular Matrix occured: \n')
        print('D: \n', d, '\n F: \n', f, '\n')
        print('WMatrix: \n', W)
        Q = al.inv(W)  # trying different inversion method

    b = np.append(c0*W10, np.zeros(N-1))  # extra vector for open BCs
    Qb = np.dot(Q, b)  # product is calculated
    T = al.expm(W)  # storing exponential matrix

    # computing residual vector for mucus model
    n = int(sp.binom(M, 2))  # number of combinations for different c-profiles
    RR = np.zeros((n, N))

    k = 0
    for i in range(M):
        for j in range(M):
            if j > i:
                RR[k, :] = cc[:, j] - fp.calcC(cc[:, i], (tt[j] - tt[i]),
                                               T=T, Qb=Qb, bc='open1side')
                k += 1

    # calculating vector of residuals
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations

    # print out error estimate in form of standart deviation if wanted
    if (verb):
        E = np.sqrt(np.sum(RRn**2)/(N*n))  # normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, bnds, cc, tt, deltaX=1, c0=None, verb=0):

    if verb == -1:
        funcVerb = True
        scpVerb = 0
    else:
        funcVerb = False
        scpVerb = verb

    optimize = ft.partial(resFun, cc=cc, tt=tt, deltaX=deltaX, c0=c0,
                          verb=funcVerb)

    initVal = np.concatenate((DRange, FRange))
    # running freely with standart termination conditions
    result = op.least_squares(optimize, initVal, bounds=bnds,
                              max_nfev=None, verbose=scpVerb)

    return result


def main():
    # reading input and setting up analysis
    (verbosity, Runs, ana, deltaX, c0, xx, cc, tt,
     bnds, FInit, DInit) = io.startUp('fullDF')

    dim = cc[:, 0].size  # number of discretization bins

    # ---------------- option for analysis only --------------------------- #
    if ana:
        print('\nDoing analysis only.')
        res = np.load('result.npy')
        analysis(np.array(res), c0=c0, xx=xx, cc=cc, tt=tt, plot=True, per=0.1)
        print('\nPlots have been made and data was extraced and saved.')
        sys.exit()
    # ---------------- option for analysis only --------------------------- #

    results = []
    for i in range(Runs):
        results.append(optimization(DRange=DInit[:, i],
                                    FRange=FInit*np.ones(dim+1),
                                    bnds=bnds, cc=cc, tt=tt, deltaX=deltaX,
                                    c0=c0, verb=verbosity))
        np.save('result.npy', np.array(results))

    analysis(np.array(results), c0=c0, xx=xx, cc=cc, tt=tt, plot=True, per=0.1)

    # returns number of runs in order to compute average time per run
    return Runs


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %2.f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
