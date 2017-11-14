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
import functools as ft
import scipy.optimize as op
import plottingScripts as ps
import os
import sys

startTime = time.time()


def analysis(result, c0=None, xx=None, cc=None, tt=None, plot=False, per=0.1,
             alpha=0, bc='reflective', savePath=None):
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

    N = cc[:, 0].size  # number of bins
    n = M-1  # number of combinations for different c-profiles
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
    D_pre = np.mean(np.array([result[indices[i]].x[:N]
                              for i in range(nbr)]), axis=0)
    # average over D because only D in between bins is of importance
    D = np.array([(D_pre[i] + D_pre[i+1])/2 for i in range(D_pre.size-1)])
    D = np.append(D, D[-1])  # stupid workaround for equal length in D and F
    F_pre = np.mean(np.array([result[indices[i]].x[N:]
                              for i in range(nbr)]), axis=0)
    F = F_pre - F_pre[0]

    # gathering standart deviation for top 1% of runs
    DSTD_pre = np.std(np.array([result[indices[i]].x[:N]
                                for i in range(nbr)]), axis=0)
    # average over D because only D in between bins is of importance
    DSTD = np.array([(DSTD_pre[i] + DSTD_pre[i+1])/2
                     for i in range(DSTD_pre.size-1)])
    DSTD = np.append(DSTD, DSTD[-1])  # fixing same length in D and F
    FSTD = np.std(np.array([result[indices[i]].x[N:]
                            for i in range(nbr)]), axis=0)

    # gathering best D and F for computation of profiles
    D_best_pre = result[indices[0]].x[:N]
    D_best = np.array([(D_best_pre[i] + D_best_pre[i+1])/2  # average D
                       for i in range(D_best_pre.size-1)])
    D_best = np.append(D_best, D_best[-1])
    F_best_pre = result[indices[0]].x[N:]
    F_best = F_best_pre - F_best_pre[0]

    if bc is 'open1side':
        # adding extra bin for c0
        D = np.concatenate((np.ones(1)*D[0], D))
        F = np.concatenate((np.ones(1)*F[0], F))
        D_best_pre = np.concatenate((np.ones(1)*D_best_pre[0], D_best_pre))
        F_best_pre = np.concatenate((np.ones(1)*F_best_pre[0], F_best_pre))
        D_best = np.concatenate((np.ones(1)*D_best[0], D_best))
        F_best = np.concatenate((np.ones(1)*F_best[0], F_best))
        DSTD = np.concatenate((np.ones(1)*DSTD[0], DSTD))
        FSTD = np.concatenate((np.ones(1)*FSTD[0], FSTD))

    # computing WMatrix for apropriate boundary conditions
    # using D_pre here because original D is used for WMatrix computation
    # but averaged D is physically important
    if bc is 'open1side':
        W, W10 = fp.WMatrix(D_best_pre, F_best_pre, deltaX=deltaX, bc=bc)
    else:
        W = fp.WMatrix(D_best_pre, F_best_pre, deltaX=deltaX, bc=bc)
        W10 = None

    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[:, 0], tt[j], W=W, W10=W10, c0=c0,
                               bc=bc) for j in range(tt.size)]).T
    # -------------------------- loading results --------------------------- #

    # --------------------------- saving data ------------------------------- #
    if bc is 'open1side':
        # extended xx and cc vector for open boundary condition
        xx_ext = np.concatenate((-deltaX*np.ones(1), xx))
        ccRes_ext = np.concatenate((np.ones((1, M))*c0, ccRes))
        cc_ext = np.concatenate((np.ones((1, M))*c0, cc))
    else:
        xx_ext = xx
        ccRes_ext = ccRes
        cc_ext = cc

    # header for txt file in which concentration profiles will be saved
    header_cons = ''
    for i, t in enumerate(tt):
        header_cons += ('column%i: c-profile [micro_M] for t_%i = %i min\n'
                        % (i+2, i, int(t/60)))
    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'concentrationRes.txt', np.c_[xx_ext, ccRes_ext],
               delimiter=',',
               header=('Numerically computed concentration profiles\n'
                       'column1: x-distance [micro_m]\n'+header_cons))
    # saving averaged DF
    np.savetxt(savePath+'DF_avg.txt', np.c_[xx_ext, D, DSTD, F, FSTD],
               delimiter=',',
               header=('Diffusivity and free energy profiles from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: average diffusivity [micro_m^2/s]\n'
                       'cloumn3: stdev of diffusivity [+/- micro_m^2/s]\n'
                       'cloumn4: average free energy [k_BT]\n'
                       'cloumn5: stdev of free energy [+/- k_BT]'))
    # saving best DF
    np.savetxt(savePath+'DF_best.txt', np.c_[xx_ext, D_best, F_best],
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
        ps.plotCon(xx_ext, cc_ext, ccRes_ext, tt, locs=[1, 3], save=True,
                   path=savePath)
        # plotting averaged D and F
        ps.plotDF(xx_ext, D, F, D_STD=DSTD, F_STD=FSTD, save=True,
                  style='.--', path=savePath)
        # plotting best D and F
        ps.plotDF(xx_ext, D_best, F_best, save=True, style='.--', path=savePath,
                  name='bestDF')

    # ---------------------- regularization ------------------------------ #
    # NOTE: this is for analysis of parameter for L2 regularization
    RR = np.zeros((n, N))
    k = 0
    for j in range(1, M):
        RR[k, :] = cc[:, j] - fp.calcC(cc[:, 0], (tt[j] - tt[0]), W=W, W10=W10,
                                       c0=c0, bc='open1side')
        k += 1
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations
    residual = np.sum(RRn**2)

    # NOTE: now doing tykhonov regularization, but with smoothing
    d0 = np.roll(D_pre, -1)  # enforcing smoothness of solution
    f0 = np.roll(F_pre, -1)
    df0 = np.concatenate((d0[:-1], f0[:-1]))  # last value cannot be smoothed
    df = np.concatenate((D_pre[:-1], F_pre[:-1]))
    regularization = np.sum((df-df0)**2)

    print('\nBest solution has residuals\n|A*x-b|^2 = %f\n|x-x_0|^2 = %f' %
          (residual, regularization))
    np.savetxt(savePath+'res_alpha=%f.txt' % alpha,
               np.array([residual, regularization]),
               header='row 1: |A*x-b|^2\nrow 2: alpha*|x-x_0|^2')
    # ---------------------- regularization ------------------------------ #


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, tt, deltaX=1, c0=None, verb=False, bc='reflective',
           alpha=0):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include:
    discretization width: deltaX,
    concentration at left boundary: c0,
    regularization parameter: alpha
    '''

    M = cc[0, :].size  # number of concentration profiles
    N = cc[:, 0].size  # number of bins

    # N parameters to be optimized for complete D and F profile
    d = df[:N]
    f = df[N:]  # letting F completely free

    # calculating W and T matrix and extra variables for open BCs
    if bc is 'open1side':
        # additionall d and f for bin where c0 is constant
        # CHANGED: now take D and F at first bin same as next one
        d = np.concatenate((np.ones(1)*d[0], d))
        f = np.concatenate((np.ones(1)*f[0], f))
        W, W10 = fp.WMatrix(d, f, deltaX, bc=bc)
        try:
            Q = la.inv(W)  # inverse of W
        except la.linalg.LinAlgError:
            print('Values for which singular Matrix occured: \n')
            print('D: \n', d, '\n F: \n', f, '\n')
            print('WMatrix: \n', W)
            Q = al.inv(W)  # trying different inversion method
        b = np.append(c0*W10, np.zeros(N-1))  # extra vector for open BCs
        Qb = np.dot(Q, b)  # product is calculated
    else:
        W = fp.WMatrix(d, f, deltaX, bc=bc)
        Qb = None  # no inverse W-Matrix needed for reflective boundaries
        # and also not possible because WMatrix is singular for reflective BCs
        # testing conservation of concentration for reflective boundaries
        if np.max(np.sum(W, 0)) > 0.01:
            print("WMatrix column sum does not vanish:\n",
                  np.max(np.sum(W, 0)))
            sys.exit()

    T = al.expm(W)  # storing exponential matrix

    # computing residual vector
    # CHANGED: now only computing profiles from c at tt[0]
    n = M-1  # number of combinations for different c-profiles
    RR = np.zeros((n, N))

    k = 0
    for j in range(1, M):
        RR[k, :] = cc[:, j] - fp.calcC(cc[:, 0], (tt[j] - tt[0]), T=T, Qb=Qb,
                                       bc=bc)
        k += 1

    # calculating vector of residuals
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations

    # NOTE: now doing tykhonov regularization, but with smoothing
    d0 = np.roll(d, -1)  # enforcing smoothness of solution
    f0 = np.roll(f, -1)
    df0 = np.concatenate((d0[:-1], f0[:-1]))  # last value cannot be smoothed
    df_trunc = np.concatenate((d[:-1], f[:-1]))
    regularization = alpha*(df_trunc-df0)
    RRn = np.append(RRn, regularization)  # appended residual vector

    # print out error estimate in form of standart deviation if wanted
    if (verb):
        E = np.sqrt(np.sum(RRn**2)/(N*n))  # normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, bnds, cc, tt, deltaX=1, c0=None, verb=0,
                 bc='reflective', alpha=0):
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
                          verb=funcVerb, bc=bc, alpha=alpha)

    initVal = np.concatenate((DRange, FRange))
    # running freely with standart termination conditions
    result = op.least_squares(optimize, initVal, bounds=bnds,
                              max_nfev=None, verbose=scpVerb)

    return result


def main():
    # reading input and setting up analysis
    (bc_mode, dim, verbosity, Runs, ana, deltaX, c0, xx, cc, tt, bnds, FInit,
     DInit, alpha) = io.startUp()

    # discretization widths in stratum corneum
    dx_SC = array([0.24398598,  2.01909017,  1.83566884,  3.89236138,
                   5.48539896, 3.407229])
    # length of the different segments for computation
    x_tot = 600  # total length of system in µm
    x_1 = 200  # length of segment 1 - gel, x_1 = 200µm
    x_2 = np.sum(dx_SC)  # length of segment 2 - SC, from discretization vector
    x_3 = 400 - x_2  # length of last segment 3, total sample x_2+x_3 = 400 µm

    # defining different discretization widths, in segment_2 given by dx_SC
    dx1 = (x_1-2.5*dx_SC[0])/3.5  # discretization width in segment x_1
    # NOTE: discretizing segment 1 first 4 bins each at a distance of dx1
    # and next two bins with a distance between them of dx_SC[0]
    # same for segment 3
    dx3 = (x_3-2.5*dx_SC[-1])/3.5  # discretization width in segment x_3
    # NOTE: discretizing segment 1 first 4 bins each at a distance of dx1
    # and next two bins with a distance between them of dx_SC[0]
    deltaXX = [dx1, dx2]

     # vectors for distance between bins dxx_dist and bin width dxx_width
     # dxx_dist contains distance to previous bin, at first bin same dx is taken
     dxx_dist = np.concatenate((np.ones(4)*dx1,  # used for WMatrix
                                np.ones(2+dim+4)*dx2))
     # this vector contains width of individual bins
     dxx_width = np.concatenate((np.ones(3)*dx1, np.ones(1)*(dx1+dx2)/2,
                                 np.ones(2+dim+3)*dx2))  # used for concentration

    # ---------------- option for analysis only --------------------------- #
    if ana:
        print('\nDoing analysis only.')
        res = np.load('result.npy')
        print('Overall %i runs have been performed.' % res.size)
        analysis(np.array(res), bc=bc_mode, c0=c0, xx=xx, cc=cc, tt=tt,
                 alpha=alpha, plot=True, per=0.1)
        print('\nPlots have been made and data was extraced and saved.')
        sys.exit()
    # ---------------- option for analysis only --------------------------- #

    results = []
    for i in range(Runs):
        print('\nNow at run %i out of %i...\n' % (i+1, Runs))
        try:
            results.append(optimization(DRange=DInit[:, i],
                                        FRange=FInit*np.ones(dim),
                                        bnds=bnds, cc=cc, tt=tt, deltaX=deltaX,
                                        c0=c0, verb=verbosity, bc=bc_mode,
                                        alpha=alpha))
            np.save('result.npy', np.array(results))
        except KeyboardInterrupt:
            print('\n\nScript has been terminated.\nData will now be analyzed...')
            break

    analysis(np.array(results), bc=bc_mode, c0=c0, xx=xx, cc=cc, tt=tt,
             alpha=alpha, plot=True, per=0.1)

    # returns number of runs in order to compute average time per run
    return Runs


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %.2f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
