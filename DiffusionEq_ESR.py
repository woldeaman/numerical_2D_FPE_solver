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


def analysis(result, dx_dist, c0=None, xx=None, cc=None, tt=None, plot=False,
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
    if plot:
        M = tt.size  # number of different profiles

    N = cc[:, 0].size  # number of bins
    n = M-1  # number of combinations for different c-profiles
    if xx is None:
        xx = np.arange(N)
    # ----------------- setting working parameters --------------------- #

    # -------------------------- loading results --------------------------- #
    # gathering data from simulations

    # loading error values, factor two, because of cost function definition
    Error = np.array([np.sqrt(2*result[i].cost / (n*N))
                      for i in range(I)])
    indices = np.argsort(Error)  # for sorting according to error

    # NOTE: only three parameters for D,F in gel, SC, and sub-SC
    # gathering mean of F and D for best 1% of runs
    D = np.mean(np.array([result[indices[i]].x[:3]
                          for i in range(nbr)]), axis=0)
    F_pre = np.mean(np.array([result[indices[i]].x[3:]
                              for i in range(nbr)]), axis=0)
    F = F_pre - F_pre[0]

    # gathering standart deviation for top 1% of runs
    DSTD = np.std(np.array([result[indices[i]].x[:3]
                            for i in range(nbr)]), axis=0)
    FSTD = np.std(np.array([result[indices[i]].x[3:]
                            for i in range(nbr)]), axis=0)

    # gathering best D and F for computation of profiles
    D_best = result[indices[0]].x[:3]
    F_best_pre = result[indices[0]].x[3:]
    F_best = F_best_pre - F_best_pre[0]

    # computing full DF profile
    segments = np.concatenate((np.zeros(6), np.ones(N-2), np.ones(6)*2)).astype(int)
    d, f = fp.computeDF(D_best, F_best, shape=segments)
    W = fp.WMatrixVarESR(d, f, deltaXX=dx_dist, con=True)

    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[:, 0], tt[j], W=W) for j in range(tt.size)]).T
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
    np.savetxt(savePath+'DF_avg.txt', np.c_[xx, D, DSTD, F, FSTD],
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
        ps.plotCon(xx, cc, ccRes, tt, locs=[1, 3], save=True,
                   path=savePath)
        # plotting averaged D and F
        ps.plotDF(xx, D, F, D_STD=DSTD, F_STD=FSTD, save=True,
                  style='.--', path=savePath)
        # plotting best D and F
        ps.plotDF(xx, D_best, F_best, save=True, style='.--', path=savePath,
                  name='bestDF')


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, tt, deltaX=1, dx_dist=None, dx_width=None,
           c0=None, verb=False, bc='reflective', alpha=0):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include:
    discretization width: deltaX,
    concentration at left boundary: c0,
    regularization parameter: alpha
    '''

    M = cc[0, :].size  # number of concentration profiles
    N = cc[:, 0].size  # number of bins

    # 3 parameters to be optimized for D,F in gel, SC, and sub-SC
    dPre = df[:3]
    fPre = df[3:]  # letting F completely free

    segments = np.concatenate((np.zeros(6), np.ones(6), np.ones(6)*2)).astype(int)
    d, f = fp.computeDF(dPre, fPre, shape=segments)

    W = fp.WMatrixVarESR(d, f, deltaXX=dx_dist, con=True)
    # QUESTION: this matrix seems to be highly non-conservative, due to
    # the extremely different values for the dx in the SC-segment!
    # --> what can we do about this?

    # testing conservation of concentration for reflective boundaries
    if abs(np.sum((np.sum(W, 0)))) > 0.01:
        print("WMatrix total sum does not vanish!\nMax value of column sum is:",
              np.max(abs(np.sum(W, 0))), '\nFor each column:\n', np.sum(W, 0))
        sys.exit()

    # testing conservation of concentration
    con = np.sum(cc[:, 0]*dx_width)
    # compute profiles from c0 and do the same conservation check
    ccComp = [fp.calcC(cc[:, 0], t=tt[i], W=W) for i in range(M)]

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
    RR = np.zeros((n, N))

    k = 0
    for j in range(1, M):
        RR[k, :] = cc[:, j] - fp.calcC(cc[:, 0], (tt[j] - tt[0]), W=W)
        k += 1

    # calculating vector of residuals
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations

    # # NOTE: now doing tykhonov regularization, but with smoothing
    # d0 = np.roll(d, -1)  # enforcing smoothness of solution
    # f0 = np.roll(f, -1)
    # df0 = np.concatenate((d0[:-1], f0[:-1]))  # last value cannot be smoothed
    # df_trunc = np.concatenate((d[:-1], f[:-1]))
    # regularization = alpha*(df_trunc-df0)
    # RRn = np.append(RRn, regularization)  # appended residual vector

    # print out error estimate in form of standart deviation if wanted
    if (verb):
        E = np.sqrt(np.sum(RRn**2)/(N*n))  # normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, bnds, cc, tt, deltaX=1, dx_dist=None,
                 dx_width=None, c0=None, verb=0, bc='reflective', alpha=0):
    """
    Helper function for non-linear LS optimization to profiles.
    """

    if verb == -1:
        funcVerb = True
        scpVerb = 0
    else:
        funcVerb = False
        scpVerb = verb

    optimize = ft.partial(resFun, cc=cc, tt=tt, deltaX=deltaX,
                          dx_dist=dx_dist, dx_width=dx_width, c0=c0,
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
    dx_SC = np.array([0.24398598,  2.01909017,  1.83566884,  3.89236138,
                      5.48539896, 3.407229])
    # from this: get distance between bins
    dx_SC_dist = np.array([(dx_SC[i]+dx_SC[i-1])/2 for i in range(1, dx_SC.size)])
    dx_SC_dist = np.append(dx_SC[0:1], dx_SC_dist)  # first bin has same dx

    # length of the different segments for computation
    x_1 = 200  # length of segment 1 - gel, x_1 = 200µm
    x_2 = np.sum(dx_SC)  # length of segment 2 - SC, from discretization vector
    x_3 = 400 - x_2  # length of last segment 3, total sample x_2+x_3 = 400 µm

    # defining different discretization widths, in segment_2 given by dx_SC
    dx1 = (x_1-3.5*dx_SC_dist[0])/2.5  # discretization width in segment x_1
    # NOTE: discretizing segment 1 first 4 bins each at a distance of dx1
    # and next 2 bins with a distance between them of dx_SC[0]
    # same for segment 3
    dx3 = (x_3-2.5*dx_SC_dist[-1])/3.5  # discretization width in segment x_3
    # NOTE: discretizing segment 3 first 3 bins each at a distance of dx_SC[-1]
    # and next 3 bins with a distance between them of dx3

    # vectors for distance between bins dxx_dist and bin width dxx_width
    # dxx_dist contains distance to previous bin, at first bin same dx is taken
    dxx_dist = np.concatenate((np.ones(3)*dx1,  # used for WMatrix
                               np.ones(3)*dx_SC_dist[0], dx_SC_dist,
                               np.ones(3)*dx_SC_dist[-1], np.ones(4)*dx3))
    # append one dx to the end, because of W-Matrix computation (needs dx[i+1])
    # this vector contains width of individual bins
    dxx_width = np.concatenate((np.ones(2)*dx1,  # used for concentration
                                np.ones(1)*(dx1+dx_SC[0])/2,
                                np.ones(3)*dx_SC[0],
                                dx_SC, np.ones(2)*dx_SC[-1],
                                np.ones(1)*(dx3+dx_SC[-1])/2, np.ones(3)*dx3))

    xx = np.arange(dxx_width.size)  # custom x-vector for plotting
    # ---------------- option for analysis only --------------------------- #
    if ana:
        print('\nDoing analysis only.')
        res = np.load('result.npy')
        print('Overall %i runs have been performed.' % res.size)
        analysis(np.array(res), dx_dist=dxx_dist, dx_width=dxx_width,
                 bc=bc_mode, c0=c0, xx=xx, cc=cc, tt=tt,
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
                                        bnds=bnds, cc=cc, tt=tt,
                                        dx_dist=dxx_dist, dx_width=dxx_width,
                                        deltaX=deltaX,
                                        c0=c0, verb=verbosity, bc=bc_mode,
                                        alpha=alpha))
            np.save('result.npy', np.array(results))
        except KeyboardInterrupt:
            print('\n\nScript has been terminated.\nData will now be analyzed...')
            break

    analysis(np.array(results), bc=bc_mode, c0=c0, xx=xx, cc=cc, tt=tt,
             dx_dist=dxx_dist, dx_width=dxx_width, alpha=alpha, plot=True,
             per=0.1)

    # returns number of runs in order to compute average time per run
    return Runs


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %.2f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
