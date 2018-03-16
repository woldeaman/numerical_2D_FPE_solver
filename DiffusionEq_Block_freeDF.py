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


def analysis(result, dx_dist, dfParams=None, dx_width=None, c0=None,
             xx=None, cc=None, tt=None, deltaX=None, plot=False, per=0.1, alpha=0,
             bc='reflective', savePath=None):
    '''
    Function analyses results from ls-optimization,
    if given comparison plots of results and original concentration profiles
    cc[i, :, :] at time tt[i] will be made, where D and F is averaged over
    top 'per' percent (standart is 0.1 - averaged over top 10%)
    '''

    # ----------------- setting working parameters --------------------- #
    # saving output in current results folder in current directory
    if savePath is None:
        savePath = os.path.join(os.getcwd(), 'results_freeDF/')
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    I = result.size  # number of different ls-opti runs
    nbr = np.ceil(per*I).astype(int)  # number for top x% of the runs
    M = len(cc)

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
    D_pre = np.mean(np.array([result[indices[i]].x[:N]
                              for i in range(nbr)]), axis=0)
    F_pre = np.mean(np.array([result[indices[i]].x[N:]
                              for i in range(nbr)]), axis=0)

    # average D and F
    D = np.array([(D_pre[i] + D_pre[i+1])/2 for i in range(D_pre.size-1)])
    D = np.append(D, D[-1])  # stupid workaround for equal length in D and F
    F_pre = np.mean(np.array([result[indices[i]].x[N:]
                              for i in range(nbr)]), axis=0)
    F = F_pre - F_pre[0]
    # for extrapolation in bulk
    D_mean = np.concatenate((np.ones(6)*D[0], D))
    F_mean = np.concatenate((np.ones(6)*F[0], F))

    # gathering standart deviation for top x% of runs
    # computed from gauß errorpropagation for errors of D1, D2 or F1, F2
    # contribution of t_D, d_D, t_F and d_F neglected for now
    DSTD_pre = np.std(np.array([result[indices[i]].x[:N]
                                for i in range(nbr)]), axis=0)
    FSTD_pre = np.std(np.array([result[indices[i]].x[N:]
                                for i in range(nbr)]), axis=0)
    # for extrapolation in bulk
    DSTD = np.concatenate((np.ones(6)*DSTD_pre[0], DSTD_pre))
    FSTD = np.concatenate((np.ones(6)*FSTD_pre[0], FSTD_pre))

    # gathering best D and F for computation of profiles
    D_best_pre = result[indices[0]].x[:N]
    D_best_pre = np.array([(D_best_pre[i] + D_best_pre[i+1])/2
                           for i in range(D_best_pre.size-1)])
    # stupid workaround for equal length in D and F
    D_best_pre = np.append(D_best_pre, D_best_pre[-1])
    F_best_preNoGauge = result[indices[0]].x[N:]
    F_best_pre = F_best_preNoGauge - F_best_preNoGauge[0]

    # for extrapolation in bulk
    D_best = np.concatenate((np.ones(6)*D_best_pre[0], D_best_pre))
    F_best = np.concatenate((np.ones(6)*F_best_pre[0], F_best_pre))

    # computing rate matrix
    W = fp.WMatrixVar(D_best, F_best, start=4, end=None, deltaXX=dx_dist,
                      con=True)

    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[0], tt[j], W=W, bc=bc) for j in range(M)]).T
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
    # for labeling the x-axis correctly
    xlabels = [[xx[0]]+[x for x in xx[6::5]],
               [-1350]+[i*50 for i in range(xx[6::5].size)]]
    if plot:
        # plotting profiles
        ps.plotConSkin(xx, cc, ccRes, tt, locs=[1, 3], save=True, path=savePath,
                       end=None, xticks=xlabels)
        # plotting averaged D and F
        ps.plotDF(xx, D_mean, F_mean, D_STD=DSTD, F_STD=FSTD, save=True,
                  style='.--', path=savePath, xticks=xlabels)
        ps.plotDF(xx, D_best, F_best, save=True, style='.--', name='bestDF',
                  path=savePath, xticks=xlabels)


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, xx, tt, deltaX=1, dx_dist=None, dx_width=None,
           c0=None, verb=False, alpha=0, bc='reflective'):
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
    d = df[:N]
    f = df[N:]  # letting F completely free

    # for extrapolation into bulk assume same d and f
    d = np.concatenate((np.ones(6)*d[0], d))
    f = np.concatenate((np.ones(6)*f[0], f))

    # computing matrix with variable discretization
    # start needs to be smaller than 6, because D, F is const. only there
    W = fp.WMatrixVar(d, f, start=4, end=None, deltaXX=dx_dist, con=True)

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

    # computing residual vector
    n = M-1  # number of combinations for different c-profiles

    # NOTE: adding additional source term in bulk, laser light depletion
    # only substract concentrations in bulk, first 21 bins, until 200µm
    # source_term = np.concatenate((np.ones(21), np.zeros(cc[1].size-21)*k))

    RR = np.array([cc[j] - ccComp[j][6:] for j in range(1, M)]).T

    # calculating vector of residuals
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations

    # print out error estimate in form of standart deviation if wanted
    if (verb):
        E = np.sqrt(np.sum(RRn**2)/(N*n))  # normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, bnds, cc, tt, xx, deltaX=1, c0=None, alpha=0,
                 dx_dist=None, dx_width=None, verb=0, bc='reflective'):
    """
    Helper function for non-linear LS optimization to profiles.
    """

    if verb == -1:
        funcVerb = True
        scpVerb = 0
    else:
        funcVerb = False
        scpVerb = verb

    optimize = ft.partial(resFun, cc=cc, tt=tt, deltaX=deltaX, c0=c0, xx=xx,
                          bc=bc, dx_dist=dx_dist, dx_width=dx_width,
                          verb=funcVerb, alpha=alpha)

    initVal = np.concatenate((DRange, FRange))
    # running freely with standart termination conditions
    result = op.least_squares(optimize, initVal, bounds=bnds,
                              max_nfev=None, verbose=scpVerb)

    return result


def main():
    # reading input and setting up analysis
    (bc_mode, dim, verbosity, Runs, ana, deltaX, c0, xx, cc, tt, bnds, FInit,
     DInit, alpha) = io.startUp()

    # ------------------------- discretization ------------------------ #
    # lenght of the different segments for computation
    x_tot = 1780  # total length of system in µm
    x_2 = np.max(xx)  # length of segment 2
    x_1 = x_tot - x_2  # length of segment 1

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
                               np.ones(2+dim+1)*dx2))
    # this vector contains width of individual bins
    dxx_width = np.concatenate((np.ones(3)*dx1, np.ones(1)*(dx1+dx2)/2,
                                np.ones(2+dim)*dx2))  # used for concentration
    # NOTE:
    # dxx_dist has one element more than dxx_width because it for WMatrix
    # computation dx at i+1 is necccessary --> needed for last bin too
    # ------------------------- discretization ------------------------ #

    # NOTE: building c0 profile, assume c0 const. in bulk
    c_const = 1  # normalized to bulk concentration c0=1
    c0 = cc[:, 0]
    c0 = np.concatenate((np.ones(6)*c_const, c0))
    cc = [c0] + [cc[:, i] for i in range(1, cc[0, :].size)]  # now with c0

    # custom x-vector, only for analysis and plotting
    xx = np.arange(c0.size)

    # ---------------- option for analysis only --------------------------- #
    if ana:
        print('\nDoing analysis only.')
        res = np.load('result.npy')
        print('Overall %i runs have been performed.' % res.size)
        analysis(np.array(res), bc=bc_mode, c0=c0, xx=xx, cc=cc, tt=tt,
                 deltaX=deltaXX, alpha=alpha, plot=True, per=0.1,
                 dx_dist=dxx_dist, dx_width=dxx_width)
        print('\nPlots have been made and data was extraced and saved.')
        sys.exit()
    # ---------------- option for analysis only --------------------------- #

    results = []
    for i in range(Runs):
        print('\nNow at run %i out of %i...\n' % (i+1, Runs))
        try:
            results.append(optimization(DRange=DInit[:, i],
                                        FRange=FInit*np.ones(dim), dx_dist=dxx_dist,
                                        dx_width=dxx_width, bnds=bnds, cc=cc,
                                        tt=tt, deltaX=deltaXX, c0=c0, xx=xx,
                                        verb=verbosity, bc=bc_mode,
                                        alpha=alpha))
            np.save('result.npy', np.array(results))
        except KeyboardInterrupt:
            print('\n\nScript has been terminated.\nData will now be analyzed...')
            break

    analysis(np.array(results), bc=bc_mode, c0=c0, xx=xx, cc=cc,
             tt=tt, deltaX=deltaXX, alpha=alpha, plot=True,
             per=0.1, dx_dist=dxx_dist, dx_width=dxx_width)

    # returns number of runs in order to compute average time per run
    return Runs


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %.2f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
