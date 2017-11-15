# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# # use this for matplotlib on the cluster
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import time
import inputOutput as io
import FPModel as fp
import functools as ft
import scipy.optimize as op
import plottingScripts as ps
import os
import sys

startTime = time.time()


def analysis(result, xx_DF, dx_dist, dfParams=None, dx_width=None, c0=None,
             tInt=None, xx=None, cc=None, tt=None, deltaX=None, plot=False,
             per=0.1, alpha=0, bc='reflective', savePath=None):
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
    M = len(cc)  # number of different profiles

    N = cc[1].size  # number of bins
    n = M-1  # number of combinations for different c-profiles

    if xx is None:
        xx = np.arange(N)
    if deltaX is None:
        deltaX = abs(xx[0] - xx[1])
    # ----------------- setting working parameters --------------------- #

    # -------------------------- loading results --------------------------- #
    # gathering data from simulations

    # loading error values, factor two, because of cost function definition
    Error = np.array([np.sqrt(2*result[i].cost / (n*N)) for i in range(I)])
    indices = np.argsort(Error)  # for sorting according to error

    # gathering mean of F and D for best x% of runs
    D_pre = np.mean(np.array([result[indices[i]].x[:dfParams]
                              for i in range(nbr)]), axis=0)
    F_preNoGauge = np.mean(np.array([result[indices[i]].x[dfParams:dfParams*2]
                                     for i in range(nbr)]), axis=0)
    F_pre = F_preNoGauge - F_preNoGauge[0]
    # computing step function d and f profiles
    D_mean = fp.stepDF(D_pre, tInt, xx_DF)
    F_mean = fp.stepDF(F_pre, tInt, xx_DF)

    # gathering standart deviation for top x% of runs
    # computed from gauß errorpropagation for errors of D1, D2 or F1, F2
    # contribution of t_D, d_D, t_F and d_F neglected for now
    DSTD_pre = np.std(np.array([result[indices[i]].x[:dfParams]
                                for i in range(nbr)]), axis=0)
    FSTD_pre = np.std(np.array([result[indices[i]].x[dfParams:dfParams*2]
                                for i in range(nbr)]), axis=0)
    # computing step function d and f profiles
    DSTD = fp.stepDF(DSTD_pre, tInt, xx_DF)
    FSTD = fp.stepDF(FSTD_pre, tInt, xx_DF)

    # gathering best D and F for computation of profiles
    D_best_pre = result[indices[0]].x[:dfParams]
    F_best_preNoGauge = result[indices[0]].x[dfParams:dfParams*2]
    F_best_pre = F_best_preNoGauge - F_best_preNoGauge[0]
    # computing step function d and f profiles
    D_best = fp.stepDF(D_best_pre, tInt, xx_DF)
    F_best = fp.stepDF(F_best_pre, tInt, xx_DF)

    # computing rate matrix
    W = fp.WMatrixVar(D_best, F_best,  start=4, end=None,
                      deltaXX=dx_dist, con=True)

    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[0], tt[j], W=W, bc=bc)
                      for j in range(M)]).T

    # computing smoothed profiles as for residuals
    cc_avg = np.array([np.array([np.average([ccRes[i, j], ccRes[i+1, j]]) if i == 0 else
                       np.average([ccRes[i-1, j], ccRes[i, j]]) if i == ccRes[:, j].size-1
                       else np.average([ccRes[i-1, j], ccRes[i, j], ccRes[i+1, j]])
                       for i in range(ccRes[:, j].size)]) for j in range(ccRes[0, :].size)]).T
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
    np.savetxt(savePath+'DF_avg.txt', np.c_[xx, D_mean, DSTD, F_mean, FSTD],
               delimiter=',',
               header=('Diffusivity and free energy profiles from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: average diffusivity [micro_m^2/s]\n'
                       'cloumn3: stdev of diffusivity [+/- micro_m^2/s]\n'
                       'cloumn4: average free energy [k_BT]\n'
                       'cloumn5: stdev of free energy [+/- k_BT]'))
    # saving best DF
    np.savetxt(savePath+'DF_best.txt', np.c_[xx, D_best, F_best],
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
        ps.plotConSkin(xx, cc, ccRes, tt, locs=[1, 3], save=True, path=savePath,
                       name='profiles_raw')
        ps.plotConSkin(xx, cc, cc_avg, tt, locs=[1, 3], save=True,
                       path=savePath, name='profiles_smoothed')

        # plotting averaged D and F
        ps.plotDF(xx, D_mean, F_mean, D_STD=DSTD, F_STD=FSTD, save=True,
                  style='.--', path=savePath)
        ps.plotDF(xx, D_best, F_best, save=True, style='.--', name='bestDF',
                  path=savePath)


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, xx, tt, dfParams, tInt=None, deltaX=1, dx_dist=None,
           dx_width=None, c0=None, verb=False, bc='reflective', alpha=0):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include:
    discretization width: deltaX,
    concentration at left boundary: c0,
    regularization parameter: alpha
    number of different dfParameters: dfParams
    for variable discretization -
    distance beteween bins: dx_dist
    width of individual bins: dx_width
    '''

    M = len(cc)  # number of concentration profiles, additional c0 profile
    N = cc[1].size  # number of bins

    # gathering D and F from non LSQ algorithm
    # DF parameters to be optimized D1, D2, F1, F2, t_D, d_D, t_F, d_F
    d = df[:dfParams]
    f = df[dfParams:dfParams*2]  # letting F completely free

    # computing step function d and f profiles
    D = fp.stepDF(d, tInt, xx)
    F = fp.stepDF(f, tInt, xx)

    # # gathering c0 profile and t0 from non LSQ algorithm
    # c0_const = df[-2]
    # gel = 14  # number of bins for gel
    # bulk = dx_width.size - gel  # number of bins for bulk
    # c0 = np.concatenate((np.ones(bulk)*c0_const, np.zeros(gel)))
    # cc = [c0] + cc  # add c0 profile to list of all profiles
    # t0 = df[-1]
    # shift time vector, first profile at t=0, next t0 later, then every 3min

    # computing matrix with variable discretization
    # start needs to be smaller than 6, because D, F is const. only there
    W = fp.WMatrixVar(D, F,  start=4, end=None, deltaXX=dx_dist, con=True)

    # testing conservation of concentration for reflective boundaries
    # NOTE: Column sum does not vanish anymore for variable binning, but equally
    # large positive and negative terms appear at binning transition --> total sum zeros
    if abs(np.sum((np.sum(W, 0)))) > 0.01:
        print("WMatrix total sum does not vanish!\nMax is:",
              np.max(np.sum(W, 0)), '\nFor each column:\n', np.sum(W, 0))
        sys.exit()

    # testing conservation of concentration
    con = np.sum(cc[0]*dx_width)
    # compute profiles from c0 and do the same conservation check
    ccComp = [fp.calcC(cc[0], t=tt[i], W=W) for i in range(M)]

    if np.any(np.array([abs(np.sum(ccComp[i]*dx_width)-con)
                        for i in range(M)]) > 0.01*con):
            print('Error: Computed concentration '
                  'is not conserved in profiles: \n',
                  np.nonzero(np.array([abs(np.sum(ccComp[i]*dx_width)-con)
                                       for i in range(M)]) > 0.01*con))
            print([np.sum(ccComp[i]*dx_width) for i in range(M)], '\n')
            print('concentration:\n', con)
            print('WMatrix Size:\n', W.shape)
            print('WMatrix Row Sum:\n', np.sum(W, 0))
            print('WMatrix 2Sum:\n', np.sum(np.sum(W, 0)))
            sys.exit()

    # T = al.expm(W)  # storing exponential matrix
    # computing residual vector
    n = M-1  # number of combinations for different c-profiles

    # NOTE: compare only averaged c_num to experimental values
    cc_avg = [np.array([np.average([ccComp[j][i], ccComp[j][i+1]]) if i == 0 else
                       np.average([ccComp[j][i-1], ccComp[j][i]]) if i == ccComp[j].size-1
                       else np.average([ccComp[j][i-1], ccComp[j][i], ccComp[j][i+1]])
                       for i in range(ccComp[j].size)]) for j in range(len(ccComp))]

    RR = np.array([cc[j] - cc_avg[j][6:-3] for j in range(1, M)]).T

    # calculating vector of residuals
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations

    # # NOTE: now doing tykhonov regularization, but with smoothing
    # d0 = np.roll(D, -1)  # enforcing smoothness of solution
    # f0 = np.roll(F, -1)
    # df0 = np.concatenate((d0[:-1], f0[:-1]))  # last value cannot be smoothed
    # df_trunc = np.concatenate((D[:-1], F[:-1]))
    # regularization = alpha*(df_trunc-df0)
    # RRn = np.append(RRn, regularization)  # appended residual vector

    # print out error estimate in form of standart deviation if wanted
    if (verb):
        E = np.sqrt(np.sum(RRn**2)/(N*n))  # normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, tdRange, bnds, cc, xx, tt, dfParams=None,
                 deltaX=1, c0=None, dx_dist=None, dx_width=None,
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

    optimize = ft.partial(resFun, cc=cc, xx=xx, tt=tt, deltaX=deltaX, c0=c0,
                          bc=bc, dx_dist=dx_dist, dx_width=dx_width,
                          tInt=tdRange, verb=funcVerb, alpha=alpha,
                          dfParams=dfParams)

    initVal = np.concatenate((DRange, FRange))
    # running freely with standart termination conditions
    result = op.least_squares(optimize, initVal, bounds=bnds, max_nfev=None,
                              verbose=scpVerb)

    return result


def main():
    # reading input and setting up analysis
    (bc_mode, dim, verbosity, Runs, ana, deltaX, c0, xx, cc, tt, bnds, FInit,
     DInit, alpha) = io.startUp()

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
    c_const = cc[0, -1]  # take first value of last profile as c0
    gel = 14  # number of bins for gel
    bulk = dxx_width.size - gel  # number of bins for bulk
    c0 = np.concatenate((np.ones(bulk)*c_const, np.zeros(gel)))
    cc = [cc[:, i] for i in range(cc[0, :].size)]
    cc = [c0] + cc  # add c0 profile to list of all profiles
    t0 = 120  # time after c0 profile
    tt = np.concatenate((np.zeros(1), tt+t0)).astype(int)

    # overriding bounds for custom set of parameters
    DBound = 1000
    FBound = 20
    params = 2  # only two values for D, F

    bndsDUpper = np.ones(params)*DBound
    bndsFUpper = np.ones(params)*FBound
    bndsDLower = np.zeros(params)
    bndsFLower = np.ones(params)*(-FBound)
    # bounds for interface position: zero and max x position
    # tBoundsLower = np.zeros(1)
    # tBoundsUpper = np.ones(1)*np.max(xx)
    FInit = np.zeros(params)
    DInit = (np.random.rand(params, Runs)*DBound)
    tInit = 50  # interface at 50 µm

    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    # custom x-vector, only for analysis and plotting
    xx = np.arange(c0.size)
    # used to compute sigmoidal DF profiles, x<0 for first 6 bins
    xx_DF = np.array([np.sum(dxx_dist[6:i]) if i >= 6
                      else -np.sum(dxx_dist[i+1:7])
                      for i in range(dxx_dist.size-1)])
    # ---------------- option for analysis only --------------------------- #
    if ana:
        print('\nDoing analysis only.')
        res = np.load('result.npy')
        print('Overall %i runs have been performed.' % res.size)
        analysis(np.array(res), bc=bc_mode, c0=c0, xx=xx, xx_DF=xx_DF, cc=cc, tt=tt,
                 deltaX=deltaXX, alpha=alpha, plot=True, per=0.1, tInt=tInit,
                 dx_dist=dxx_dist, dx_width=dxx_width, dfParams=params)
        print('\nPlots have been made and data was extraced and saved.')
        sys.exit()
    # ---------------- option for analysis only --------------------------- #

    results = []
    for i in range(Runs):
        print('\nNow at run %i out of %i...\n' % (i+1, Runs))
        try:
            results.append(optimization(DRange=DInit[:, i],
                                        FRange=FInit, dfParams=params,
                                        tdRange=tInit, dx_dist=dxx_dist,
                                        dx_width=dxx_width, bnds=bnds, cc=cc,
                                        tt=tt, xx=xx_DF, c0=c0,
                                        verb=verbosity, bc=bc_mode,
                                        alpha=alpha))
            np.save('result.npy', np.array(results))
        except KeyboardInterrupt:
            print('\n\nScript has been terminated.\nData will now be analyzed...')
            break

    analysis(np.array(results), bc=bc_mode, c0=c0, xx=xx, xx_DF=xx_DF,
             tInt=tInit, cc=cc, tt=tt, deltaX=deltaXX, alpha=alpha, plot=True,
             per=0.1, dx_dist=dxx_dist, dx_width=dxx_width, dfParams=params)

    # returns number of runs in order to compute average time per run
    return Runs


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %.2f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
