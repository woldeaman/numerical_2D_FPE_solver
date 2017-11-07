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
# import numpy.linalg as la
import functools as ft
import scipy.optimize as op
import plottingScripts as ps
import os
import sys

startTime = time.time()


def analysis(result, dfParams, c0=None, xx=None, deltaX=None, cc=None,
             tt=None, plot=False, per=0.1, alpha=0, bc='reflective', savePath=None):
    '''
    Function analyses results from ls-optimization,
    if given comparison plots of results and original concentration profiles
    cc[i, :, :] at time tt[i] will be made, where D and F is averaged over
    top 'per' percent (standart is 0.1 - averaged over top 10%)
    '''

    # ----------------- setting working parameters --------------------- #
    # saving output in current results folder in current directory
    if savePath is None:
        savePath = os.path.join(os.getcwd(), 'results_alpha=%f/' % alpha)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    I = result.size  # number of different ls-opti runs
    nbr = np.ceil(per*I).astype(int)  # number for top x% of the runs
    if plot:
        M = tt.size  # number of different profiles

    N = cc[1].size  # number of bins
    n = M-1  # number of combinations for different c-profiles
    params = dfParams
    if xx is None:
        xx = np.arange(N)
    if deltaX is None:
        deltaX = abs(xx[0] - xx[1])
    # ----------------- setting working parameters --------------------- #

    # -------------------------- loading results --------------------------- #
    # gathering data from simulations

    # loading error values, factor two, because of cost function definition
    Error = np.array([np.sqrt(2*result[i].cost / (n*N))
                      for i in range(I)])
    indices = np.argsort(Error)  # for sorting according to error

    # gathering mean of F and D for best x% of runs
    D_pre = np.mean(np.array([result[indices[i]].x[:params]
                              for i in range(nbr)]), axis=0)
    # average over D because only D in between bins is of importance
    D = np.array([(D_pre[i] + D_pre[i+1])/2 for i in range(D_pre.size-1)])
    D = np.append(D, D[-1])  # stupid workaround for equal length in D and F
    F_pre = np.mean(np.array([result[indices[i]].x[params:]
                              for i in range(nbr)]), axis=0)
    F = F_pre - F_pre[0]

    # gathering standart deviation for top 1% of runs
    DSTD_pre = np.std(np.array([result[indices[i]].x[:params]
                                for i in range(nbr)]), axis=0)
    # average over D because only D in between bins is of importance
    DSTD = np.array([(DSTD_pre[i] + DSTD_pre[i+1])/2
                     for i in range(DSTD_pre.size-1)])
    DSTD = np.append(DSTD, DSTD[-1])  # fixing same length in D and F
    FSTD = np.std(np.array([result[indices[i]].x[params:]
                            for i in range(nbr)]), axis=0)

    # gathering best D and F for computation of profiles
    D_best_pre = result[indices[0]].x[:params]
    D_best = np.array([(D_best_pre[i] + D_best_pre[i+1])/2  # average D
                       for i in range(D_best_pre.size-1)])
    D_best = np.append(D_best, D_best[-1])  # for equal length in D, F
    F_best_pre = result[indices[0]].x[params:]
    F_best = F_best_pre - F_best_pre[0]

    # shaping D, F profiles
    segments = np.concatenate((np.zeros(6), np.arange(1, dfParams))).astype(int)
    d_best_pre, f_best_pre = fp.computeDF(D_best_pre, F_best_pre, shape=segments)
    d_best, f_best = fp.computeDF(D_best, F_best, shape=segments)
    d_avg, f_avg = fp.computeDF(D, F, shape=segments)
    d_std, f_std = fp.computeDF(DSTD, FSTD, shape=segments)
    # discretization widths
    dxx_DF = np.concatenate((np.ones(4)*deltaX[0], np.ones(N+6)*deltaX[1]))

    W = fp.WMatrixVar(d_best_pre, f_best_pre,  start=4, end=None, deltaXX=dxx_DF,
                      con=True)

    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[0], tt[j], W=W, bc=bc)
                      for j in range(tt.size)]).T
    # -------------------------- loading results --------------------------- #

    # --------------------------- saving data ------------------------------- #

    # header for txt file in which concentration profiles will be saved
    header_cons = ''
    for i, t in enumerate(tt):
        header_cons += ('column%i: c-profile [micro_M] for t_%i = %i min\n'
                        % (i+2, i, int(t/60)))
    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'concentrationRes.txt', np.c_[xx, ccRes],
               delimiter=',',
               header=('Numerically computed concentration profiles\n'
                       'column1: x-distance [micro_m]\n'+header_cons))
    # saving averaged DF
    np.savetxt(savePath+'DF_avg.txt', np.c_[xx, d_avg, d_std, f_avg, f_std],
               delimiter=',',
               header=('Diffusivity and free energy profiles from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: average diffusivity [micro_m^2/s]\n'
                       'cloumn3: stdev of diffusivity [+/- micro_m^2/s]\n'
                       'cloumn4: average free energy [k_BT]\n'
                       'cloumn5: stdev of free energy [+/- k_BT]'))
    # saving best DF
    np.savetxt(savePath+'DF_best.txt', np.c_[xx, d_best, f_best],
               delimiter=',',
               header=('Diffusivity and free energy profiles with lowest '
                       'error from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: diffusivity [micro_m^2/s]\n'
                       'cloumn3: free energy [k_BT]'))

    # saving Error of top 1% of runs
    np.savetxt(savePath+'minError.txt', Error[indices[:nbr]], delimiter=',',
               header=('Minimal error for top %.2f %% runs.' % (per*100)))
    # --------------------------- saving data ------------------------------- #

    # ------------------------- plotting data ------------------------------- #
    if plot:
        # plotting profiles
        ps.plotConSkin(xx, cc, ccRes, tt, locs=[1, 3], save=True, path=savePath)
        # plotting averaged D and F
        ps.plotDF(xx, d_avg, f_avg, D_STD=d_std, F_STD=f_std, save=True,
                  style='.--', path=savePath)
        ps.plotDF(xx, d_best, f_best, D_STD=d_std, F_STD=f_std, save=True,
                  style='.--', name='bestDF', path=savePath)

    # ---------------------- regularization ------------------------------ #
    RR = np.zeros((n, N))

    k = 0
    for j in range(1, M):
        # fitting only in the range where measurements where performed
        RR[k, :] = cc[j] - fp.calcC(cc[0], (tt[j] - tt[0]), W=W,
                                    bc=bc)[6:-3]
        k += 1

    # calculating vector of residuals
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations
    # NOTE/CHANGED: now doing tykhonov regularization, but with smoothing
    # QUESTION: think about last bin of D and F, currently smoothed to first bin
    df = np.concatenate((D_pre, F_pre))
    d0 = np.roll(D_pre, -1)
    f0 = np.roll(F_pre, -1)
    df0 = np.concatenate((d0, f0))

    regularization = np.sum((alpha*(df-df0))**2)
    residual = np.sum(RRn**2)
    print('\nBest solution has residuals\n|A*x-b|^2 = %f\nalpha*|x-x_0|^2 = %f' %
          (residual, regularization))
    np.savetxt(savePath+'res_alpha=%f.txt' % alpha,
               np.array([residual, regularization]),
               header='row 1: |A*x-b|^2\nrow 2: alpha*|x-x_0|^2')
    # ---------------------- regularization ------------------------------ #


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, tt, dfParams, deltaX=1, c0=None, verb=False, bc='reflective',
           alpha=0):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include:
    discretization width: deltaX,
    concentration at left boundary: c0,
    regularization parameter: alpha
    number of different dfParameters: dfParams
    '''

    M = len(cc)  # number of concentration profiles
    N = cc[1].size  # number of bins

    # complete D and F profile and one extra D', F' in bulk solution
    dPre = df[:dfParams]
    fPre = df[dfParams:]
    # first six bins are part of bulk, rest is free DF
    segments = np.concatenate((np.zeros(6), np.arange(1, dfParams))).astype(int)
    d, f = fp.computeDF(dPre, fPre, shape=segments)

    # discretization width in bulk and in measured segment
    dx1, dx2 = deltaX[0], deltaX[1]
    # this vector contains distance between previous bins, dx = x_i - x_i-1
    dxx_DF = np.concatenate((np.ones(4)*dx1,  # used for WMatrix
                             np.ones(N+6)*dx2))
    # this vector contains width of individual bins
    dxx_Con = np.concatenate((np.ones(3)*dx1, np.ones(1)*(dx1+dx2)/2,
                              np.ones(N+5)*dx2))  # used for concentration

    # computing matrix with variable discretization
    W = fp.WMatrixVar(d, f,  start=4, end=None, deltaXX=dxx_DF, con=True)

    # QUESTION: with variable binning, column sum does not vanish at transition
    # sites, there is one positive and equally negative contribution, is this right ???

    # # testing conservation of concentration for reflective boundaries
    # if np.max(np.sum(W, 0)) > 0.01:
    #     print("WMatrix column sum does not vanish!\nMax is:",
    #           np.max(np.sum(W, 0)), '\nFor each column:\n', np.sum(W, 0))
    #     sys.exit()

    # testing conservation of concentration
    con = np.sum(cc[0]*dxx_Con)
    # compute profiles from c0 and do the same conservation check
    ccComp = [fp.calcC(cc[0], t=tt[i], W=W) for i in range(M)]

    if np.any(np.array([abs(np.sum(ccComp[i]*dxx_Con)-con)
                        for i in range(M)]) > 0.01*con):
            print('Error: Computed concentration '
                  'is not conserved in profiles: \n',
                  np.nonzero(np.array([abs(np.sum(ccComp[i]*dxx_Con)-con)
                                       for i in range(M)]) > 0.01*con))
            print([np.sum(ccComp[i]*dxx_Con) for i in range(M)], '\n')
            print('concentration:\n', con)
            print('WMatrix Size:\n', W.shape)
            print('WMatrix Row Sum:\n', np.sum(W, 0))
            print('WMatrix 2Sum:\n', np.sum(np.sum(W, 0)))
            sys.exit()

    T = al.expm(W)  # storing exponential matrix
    # computing residual vector
    n = M-1  # number of combinations for different c-profiles
    RR = np.zeros((n, N))

    k = 0
    for j in range(1, M):
        RR[k, :] = cc[j] - fp.calcC(cc[0], (tt[j] - tt[0]), T=T)[6:-3]
        k += 1

    # calculating vector of residuals
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations
    # NOTE/CHANGED: now doing tykhonov regularization, but with smoothing
    # QUESTION: think about last bin of D and F, currently smoothed to first bin
    d0 = np.roll(dPre, -1)  # enforcing smoothness of solution
    f0 = np.roll(fPre, -1)
    df0 = np.concatenate((d0, f0))
    regularization = alpha*(df-df0)
    RRn = np.append(RRn, regularization)  # appended residual vector

    # print out error estimate in form of standart deviation if wanted
    if (verb):
        E = np.sqrt(np.sum(RRn**2)/(N*n))  # normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, bnds, cc, tt, dfParams, deltaX=1, c0=None,
                 verb=0, bc='reflective', alpha=0):
    """
    Helper function for non-linear LS optimization to profiles.
    """

    if verb == -1:
        funcVerb = True
        scpVerb = 0
    else:
        funcVerb = False
        scpVerb = verb

    optimize = ft.partial(resFun, cc=cc, tt=tt, deltaX=deltaX, c0=c0,
                          verb=funcVerb, bc=bc, alpha=alpha, dfParams=dfParams)

    initVal = np.concatenate((DRange, FRange))
    # running freely with standart termination conditions
    result = op.least_squares(optimize, initVal, bounds=bnds,
                              max_nfev=None, verbose=scpVerb)

    return result


def main():
    # reading input and setting up analysis
    (bc_mode, dim, verbosity, Runs, ana, deltaX, c0, xx, cc, tt, bnds, FInit,
     DInit, alpha) = io.startUp()

    # overriding bounds for custom set of parameters
    DBound = 1000
    FBound = 20
    params = dim + 4  # number of different D and F values to fit
    # one extra param for bulk and 3 extra for last three bins
    bndsDUpper = np.ones(params)*DBound
    bndsFUpper = np.ones(params)*FBound
    bndsDLower = np.zeros(params)
    bndsFLower = np.ones(params)*(-FBound)
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))
    FInit = np.zeros(params)
    DInit = (np.random.rand(params, Runs)*DBound)

    # lenght of the different segments for computation
    x_tot = 1500  # total length of system in µm
    x_2 = np.max(xx)  # length of segment 2, x_2 = 150 µm
    x_3 = 180 - x_2  # end of system is at x = 180µm, lengthx of x_3 = 30 µm
    x_1 = x_tot - (x_2 + x_3)  # length of segment 1 x_1 = 1320 µm

    # defining different discretization widths
    dx2 = deltaX  # in segment 2 and segment 3
    dx1 = (x_1-2.5*dx2)/3.5  # discretization in segment x_1
    # NOTE:
    # discretizing segment 1 first 4 bins each at a distance of dx1
    # and next two bins with a distance between them of dx2
    deltaXX = [dx1, dx2]

    # vectors for distance between bins dxx_dist and bin width dxx_width
    # dxx_dist contains distance to previous bin, at first bin same dx is taken
    dxx_dist = np.concatenate((np.ones(4)*dx1,  # used for WMatrix
                               np.ones(2+dim+4)*dx2))
    # this vector contains width of individual bins
    dxx_width = np.concatenate((np.ones(3)*dx1, np.ones(1)*(dx1+dx2)/2,
                                np.ones(2+dim+3)*dx2))  # used for concentration
    # NOTE:
    # dxx_dist has one element more than dxx_width because it for WMatrix
    # computation dx at i+1 is necccessary --> needed for last bin too


    # NOTE: building c0 profile, assume c0 const. in bulk (for x < 50 µm)
    M = cc[0, :].size  # number of profiles
    gel = 8  # number of bins for gel
    solution_dx1 = 4  # number of bins for solution with larger dx
    solution_dx2 = 13  # number of bins for solution with smaller dx

    # total amount of dextran (approximated from last profile)
    # TODO: check this
    # tot = (cc[-1][0]*132 + np.sum(cc[-1]) + cc[-1][-1]*3)*10
    # c_dx1 = (tot - cc[-1][0]*10*12)/(dx1*4)

    c_dx1 = cc[0, -1]
    c0 = np.concatenate((np.ones(solution_dx1)*c_dx1,
                         np.ones(solution_dx2)*cc[0, -1], np.zeros(gel)))
    cc = [cc[:, i] for i in range(M)]
    cc = [c0] + cc  # new cc contains all profiles in list
    tt = np.concatenate((np.zeros(1), tt+180)).astype(int)

    # custom x-vector
    xx = np.arange(c0.size)

    # ---------------- option for analysis only --------------------------- #
    if ana:
        print('\nDoing analysis only.')
        res = np.load('result.npy')
        print('Overall %i runs have been performed.' % res.size)
        analysis(np.array(res), bc=bc_mode, c0=c0, xx=xx, cc=cc, tt=tt,
                 deltaX=deltaXX, alpha=alpha, plot=True, per=0.1, dfParams=params)
        print('\nPlots have been made and data was extraced and saved.')
        sys.exit()
    # ---------------- option for analysis only --------------------------- #

    results = []
    for i in range(Runs):
        print('\nNow at run %i out of %i...\n' % (i+1, Runs))
        try:
            results.append(optimization(DRange=DInit[:, i],
                                        FRange=FInit,
                                        bnds=bnds, cc=cc, tt=tt, deltaX=deltaXX,
                                        c0=c0, verb=verbosity, bc=bc_mode,
                                        alpha=alpha, dfParams=params))
            np.save('result.npy', np.array(results))
        except KeyboardInterrupt:
            print('\n\nScript has been terminated.\nData will now be analyzed...')
            break

    analysis(np.array(results), bc=bc_mode, c0=c0, xx=xx, cc=cc, tt=tt,
             deltaX=deltaXX, alpha=alpha, plot=True, per=0.1, dfParams=params)

    # returns number of runs in order to compute average time per run
    return Runs


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %.2f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
