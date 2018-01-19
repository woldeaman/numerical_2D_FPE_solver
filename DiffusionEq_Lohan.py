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


def analysis(result, dx_dist, dfParams, c0=None, xx=None, cc=None, tt=None,
             plot=False, per=0.1, alpha=0, deltaX=None, bc='reflective',
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

    N = cc[1].size  # number of bins
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

    # for computing D and F profiles or correct shape
    segments = np.concatenate((np.zeros(6), np.ones(N), np.ones(6)*2)).astype(int)
    # gathering mean of F and D for best 1% of runs
    D = np.mean(np.array([result[indices[i]].x[:dfParams]
                          for i in range(nbr)]), axis=0)
    F_pre = np.mean(np.array([result[indices[i]].x[dfParams:]
                              for i in range(nbr)]), axis=0)
    F = F_pre - F_pre[0]
    D, F = fp.computeDF(D, F, shape=segments)

    # gathering standart deviation for top 1% of runs
    DSTD = np.std(np.array([result[indices[i]].x[:dfParams]
                            for i in range(nbr)]), axis=0)
    FSTD = np.std(np.array([result[indices[i]].x[dfParams:]
                            for i in range(nbr)]), axis=0)
    DSTD, FSTD = fp.computeDF(DSTD, FSTD, shape=segments)

    # gathering best D and F for computation of profiles
    D_best = result[indices[0]].x[:dfParams]
    F_best_pre = result[indices[0]].x[dfParams:]
    F_best = F_best_pre - F_best_pre[0]

    # computing full DF profile
    D_best, F_best = fp.computeDF(D_best, F_best, shape=segments)
    W = fp.WMatrixVar(D_best, F_best, start=3, end=(6+N+2), deltaXX=dx_dist,
                      con=True)

    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[0], tt[j], W=W) for j in range(tt.size)]).T
    # -------------------------- loading results --------------------------- #

    # --------------------------- saving data ------------------------------- #
    # for labeling the x-axis correctly
    xxPlot = np.arange(cc[0].size)
    xlabels = [[xxPlot[0], xxPlot[6], xxPlot[11], xxPlot[16], xxPlot[20], xxPlot[-1]],
               ['%i' % -200, '%i' % xx[0], '%i' % (5*deltaX+xx[0]),
                '%i' % (10*deltaX+xx[0]), '%i' % (14*deltaX+xx[0]), '%i' % 400]]

    # header for txt file in which concentration profiles will be saved
    header_cons = ''
    for i, t in enumerate(tt):
        header_cons += ('column%i: c-profile [micro_M] for t_%i = %i min\n'
                        % (i+2, i, int(t/60)))
    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'concentrationRes.txt', np.c_[xxPlot, ccRes],
               delimiter=',',
               header=('Numerically computed concentration profiles\n'
                       'column1: x-distance [micro_m]\n'+header_cons))
    # saving averaged DF
    np.savetxt(savePath+'DF_avg.txt', np.c_[xxPlot, D, DSTD, F, FSTD],
               delimiter=',',
               header=('Diffusivity and free energy profiles from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: average diffusivity [micro_m^2/s]\n'
                       'cloumn3: stdev of diffusivity [+/- micro_m^2/s]\n'
                       'cloumn4: average free energy [k_BT]\n'
                       'cloumn5: stdev of free energy [+/- k_BT]'))
    # saving best DF
    np.savetxt(savePath+'DF_best.txt', np.c_[xxPlot, D_best, F_best],
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
        ps.plotConSkin(xxPlot, cc, ccRes, tt, locs=[1, 3], save=True,
                       path=savePath, start=6, end=-6, xticks=xlabels,
                       ylabel='Concentration [nmol/cm$^2$]')
        # plotting averaged D and F
        ps.plotDF(xxPlot, D, F, D_STD=DSTD, F_STD=FSTD, save=True, xticks=xlabels,
                  style='.--', path=savePath)
        # plotting best D and F
        ps.plotDF(xxPlot, D_best, F_best, save=True, style='.--', path=savePath,
                  name='bestDF', xticks=xlabels)


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, tt, dfParams, deltaX=1, dx_dist=None, dx_width=None,
           c0=None, verb=False, bc='reflective', alpha=0):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include:
    discretization width: deltaX,
    concentration at left boundary: c0,
    regularization parameter: alpha
    dx_dist: discretization distances [for WMatrix computation]
    dx_width: discretization bin widths [for concentration computation]
    dfParams: number of parameters for D and F
    '''

    M = len(cc)  # number of concentration profiles
    N = cc[1].size  # number of bins

    # 2*3 parameters to be optimized for D,F in gel, SC, and sub-SC
    dPre = df[:dfParams]
    fPre = df[dfParams:]

    segments = np.concatenate((np.zeros(6), np.ones(N), np.ones(6)*2)).astype(int)
    d, f = fp.computeDF(dPre, fPre, shape=segments)

    W = fp.WMatrixVar(d, f, start=3, end=(6+N+2), deltaXX=dx_dist, con=True)

    # testing conservation of concentration for reflective boundaries
    if abs(np.sum((np.sum(W, 0)))) > 0.01:
        print("WMatrix total sum does not vanish!\nMax value of column sum is:",
              np.max(abs(np.sum(W, 0))), '\nFor each column:\n', np.sum(W, 0))
        sys.exit()

    # testing conservation of concentration
    con = np.sum(cc[0]*dx_width)
    # compute profiles from c0 and do the same conservation check
    # TODO: use computation for const c0 BC and add release rate vector
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

    # # value for undisrupted skin
    # c_subSC = 0.208762038  # data measured after 50 tape strips
    # value for disrupted skin
    c_subSC = 0.507484533  # data measured after 50 tape strips

    # integrated amount in subSC is to be compared with numerical profiles
    amount_subSC = c_subSC*np.sum(dx_width[-6:])

    RR = np.array([cc[i] - ccComp[i][6:-6] for i in range(1, M)]).T
    RR_subSC = amount_subSC - np.sum(ccComp[1][-6:]*dx_width[-6:])  # resiudal for sub-SC layers
    # residual vector contains all deviations
    RRn = np.append(RR, RR_subSC)
    RRn = RRn.reshape(RRn.size)

    # print out error estimate in form of standart deviation if wanted
    if (verb):
        E = np.sqrt(np.sum(RRn**2)/(N*n))  # normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, dfParams, bnds, cc, tt, deltaX=1, dx_dist=None,
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
    params = 3  # number of different D and F values to fit
    bndsDUpper = np.ones(params)*DBound
    bndsFUpper = np.ones(params)*FBound
    bndsDLower = np.zeros(params)
    bndsFLower = np.ones(params)*(-FBound)
    FInit = np.zeros(params)
    DInit = (np.random.rand(params, Runs)*DBound)
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    # TODO: change this to dimension of Lohan Experiments

    # length of the different segments for computation
    x_1 = 200  # width of cream/gel applied to skin sample
    x_2 = xx[-1] - xx[0]  # length of segment 2 - SC
    x_tot = 400  # total length of sample in µm
    x_3 = x_tot - x_2  # size of deeper skin layers below SC

    # defining different discretization widths
    dx2 = deltaX  # discretization in SC
    dx1 = (x_1-3.5*dx2)/2.5  # discretization width in gel
    dx3 = (x_3-2.5*dx2)/3.5  # discretization width in sub-SC layers

    # vectors for distance between bins dxx_dist and bin width dxx_width
    # dxx_dist contains distance to previous bin, at first bin same dx is taken
    dxx_dist = np.concatenate((np.ones(3)*dx1,  # used for WMatrix
                               np.ones(3+dim+3)*dx2, np.ones(4)*dx3))
    # append one dx to the end, because of W-Matrix computation (needs dx[i+1])
    # this vector contains width of individual bins
    dxx_width = np.concatenate((np.ones(2)*dx1,  # used for concentration
                                np.ones(1)*(dx1+dx2)/2,
                                np.ones(3+dim+2)*dx2,
                                np.ones(1)*(dx3+dx2)/2, np.ones(3)*dx3))

    # building c[t = 0] profile, at beginning everything is in gel
    c0 = 11.42
    cc0 = np.concatenate((np.ones(6)*c0, np.zeros(dim+6)))
    cc = [cc0, cc]  # new list with all Profiles
    tt = np.concatenate((np.zeros(1), 4*60*60*np.ones(1))).astype(int)

    # ---------------- option for analysis only --------------------------- #
    if ana:
        print('\nDoing analysis only.')
        res = np.load('result.npy')
        print('Overall %i runs have been performed.' % res.size)
        analysis(np.array(res), dx_dist=dxx_dist, bc=bc_mode, c0=c0, xx=xx,
                 cc=cc, tt=tt, dfParams=params, alpha=alpha, plot=True, per=0.1,
                 deltaX=dx2)
        print('\nPlots have been made and data was extraced and saved.')
        sys.exit()
    # ---------------- option for analysis only --------------------------- #

    results = []
    for i in range(Runs):
        print('\nNow at run %i out of %i...\n' % (i+1, Runs))
        try:
            results.append(optimization(DRange=DInit[:, i],
                                        FRange=FInit, dfParams=params,
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
             dx_dist=dxx_dist, alpha=alpha, plot=True,
             per=0.1, dfParams=params, deltaX=dx2)

    # returns number of runs in order to compute average time per run
    return Runs


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %.2f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
