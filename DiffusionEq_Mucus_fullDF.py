# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck --> original cervix mucus data
import numpy as np
import time
import inputOutput as io
import FPModel as fp
import scipy.linalg as al
import numpy.linalg as la
import scipy.special as sp
import functools as ft
import scipy.optimize as op
import argparse as ap
import plottingScripts as ps
import matplotlib.pyplot as plt
import os
import sys

startTime = time.time()


def analysis(result, xx=None, cc=None, tt=None, plot=False, per=0.1,
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

    N = int(result[0].x.size/2 - 1)  # number of bins, -1 because c0 - constant
    n = int(sp.binom(M, 2))  # number of combinations for different c-profiles
    if xx is None:
        xx = np.arange(N)
    deltaX = abs(xx[0] - xx[1])
    c0 = 4  # concentration of peptide solution in µM
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
        ps.plotCon(xx, cc, ccRes, tt, save=True, path=savePath)
        # plotting averaged D and F
        ps.plotDF(xx, D, F, D_STD=DSTD, F_STD=FSTD, save=True, path=savePath)


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
    # ---------------- parsing command line inputs ------------------------- #
    # gathering path to data and setting verbosity
    parser = ap.ArgumentParser()
    parser.add_argument('-p', dest='path', type=str,
                        help='define the path to data for analysis')
    parser.add_argument('-v', dest='verbosity', type=int, help='set '
                        'verbosity level ranging from 0 - no output to 2 - '
                        'full output, -1 - means custom verbose mode')
    parser.add_argument('-pre', dest='pre', action='store_true',
                        help='Only look at raw data and pre-processing'
                        'results. Does not start analysis.')

    args = parser.parse_args()
    if args.verbosity is None:
        verbosity = 0
    else:
        verbosity = args.verbosity
    # ---------------- parsing command line inputs ------------------------- #

    # -------------- reading and pre-processing profiles ------------------- #
    # reading profiles and take only samples for 4 different time points
    try:  # change seperator accordingly
        data = io.readData(args.path, sep=';')
    except ValueError:
        data = io.readData(args.path, sep=',')

    # pre processing of profiles
    # filtering and setting negative c-values to zero
    print('\nReading data and starting pre-processing.')
    xx_exp = data[:, 0]
    cc_exp = np.array([data[:, 1], data[:, 31], data[:, 61], data[:, 91]]).T
    xx, cc = io.preProcessing(xx_exp, cc_exp, order=5)
    np.savetxt('preProcessedProfiles.txt', np.c_[xx, cc], delimiter=',',
               header='Profiles were smoothed using Savitzky-Golay filter'
               ' \nCloumn 1: x-distance [micro_m]'
               '\nColumn 2-5: c-profiles at t0-t3 [micro_M]')
    print('Finished pre-processing and saved smoothed profiles.')

    if args.pre:
        # plotting smoothed and original data as comparison
        print('Raw data x-Axis ranges from Xmin = %2.f to Xmax = %2.f, '
              'with discretization deltaX = %2.f' % (np.min(xx_exp),
                                                     np.max(xx_exp),
                                                     (xx_exp[1]-xx_exp[0])))
        print('Assuming temporal discretization of deltaT = 10s we have total'
              ' data for %2.f minutes' % ((data[0, :].size-2)/6))

        # plotting profiles
        plt.plot(xx_exp, cc_exp, '--', label='original')
        plt.plot(xx, cc, '-', label='smoothed')
        plt.show()
        sys.exit()

    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 15  # concentration of peptide solution in µ
    dim = cc[:, 0].size  # number of discretization bins
    deltaX = abs(xx[0] - xx[1])  # discretization width
    print('Discretization width is %.2f µm.\n Starting optimization.\n'
          % deltaX)
    # -------------- reading and pre-processing profiles ------------------- #

    # setting reasonable bounds for F and D
    # and setting number of runs
    DBound = 1000
    FBound = 20
    Runs = 100
    # parameters is one more than number of bins, because of c0 at boundary
    # same number for D and F, because F roams freely now
    bndsDUpper = np.ones(dim+1)*DBound
    bndsFUpper = np.ones(dim+1)*FBound
    bndsDLower = np.zeros(dim+1)
    bndsFLower = np.ones(dim+1)*(-FBound)
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    FInit = 0
    DInit = (np.random.rand(dim+1, Runs)*DBound)

    results = []
    for i in range(Runs):
        results.append(optimization(DRange=DInit[:, i],
                                    FRange=FInit*np.ones(dim+1),
                                    bnds=bnds, cc=cc, tt=tt, deltaX=deltaX,
                                    c0=c0, verb=verbosity))
        np.save('result.npy', np.array(results))

    analysis(np.array(results), xx=xx, cc=cc, tt=tt, plot=True, per=0.1)

    # returns number of runs in order to compute average time per run
    return Runs


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %2.f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
