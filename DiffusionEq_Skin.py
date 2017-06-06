# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as al
import functools as ft
import scipy.optimize as op
import sys
import time
import matplotlib.pyplot as plt
import FPModel as fp
import inputOutput as io
import argparse as ap
import os


# for printing c-profiles
def plotConSkin(xx, cc, ccRes, tt, save=False, path=None, deltaXX=None):

    M = cc.size  # number of profiles
    N = ccRes[0, :].size  # number of bins
    if deltaXX is None:
        deltaXX = np.ones(N+1)
    if path is None:
        savePath = os.path.join(os.getcwd(), 'results/')
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    # plotting concentration profiles
    l1s = []  # for sperate legends
    l2s = []
    colors = ['r', 'm', 'c', 'b', 'y', 'k', 'g']

    plt.figure(0)
    for j in range(1, M):
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.xlabel('Bins')
        plt.ylabel('Concentration [µM]')
        l1, = plt.plot(xx[7:cc[j].size+7], cc[j], '--', color=colors[j])

        # plot computed only for t > 0, otherwise not computed
        l1s.append([l1])
        # concatenated to include constanc c0 boundary condition
        l2, = plt.plot(xx, ccRes[j, :], '-', color=colors[j])
        l2s.append([l2])
    # plotting two legends, for color and linestyle
    legend1 = plt.legend([l1, l2], ["Experiment", "Numerical"], loc=1)
    plt.legend([l[0] for l in l1s], ["%d min" % (tt[i]/60)
                                     for i in range(1, tt.size)], loc=2)
    plt.gca().add_artist(legend1)

    if save:
        plt.savefig(path+'profiles.pdf')
    else:
        plt.show()


def analysis(results, xx=None, cc=None, tt=None, deltaX=None, plot=False,
             per=0.1, savePath=None, fixD_derm=False, D_dermis=None):
    # ------------------- setting working parameters ----------------------- #
    if savePath is None:
        savePath = os.path.join(os.getcwd(), 'results/')
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    # max number of bins in epidermis
    N = max([cc[i].size for i in range(1, cc.size)])

    # vector of discretizations
    deltaXX = np.concatenate((np.ones(5)*deltaX[0],
                              np.ones(N+7)*deltaX[1],
                              np.ones(11)*deltaX[2]))
    # vector of different segments
    segments = np.concatenate((np.zeros(7), np.arange(1, N+1),
                               np.ones(15)*(N+1))).astype(int)
    # ------------------- experimental parameters ----------------------- #

    # -------------------------- loading results --------------------------- #

    I = results.size  # number of different initial conditions
    topPer = np.ceil(per*I).astype(int)  # number for top 1% of the runs

    # gathering data from simulations
    # loading error values
    # Error as computed by robert
    Error = np.array([np.sqrt((np.sum((results[i].fun[:73]**2)) +
                               np.sum((results[i].fun[73:153]**2)) +
                               np.sum((results[i].fun[153:233]**2))) / (3))
                      for i in range(I)])
    # Error computed from cost function value, factor 2 bc definition of cost
    # Error2 = np.array([np.sqrt((2/3)*results[i].cost) for i in range(I)])

    indices = np.argsort(Error)  # for sorting according to error
    # number of parameters for D
    DParams = N+2
    # for fixed D_dermis
    if fixD_derm:
        DParams = N+1
        # D_dermis = 67.341  # value from PNAS paper

    # gathering mean of F and D for best 1% of runs
    DRes_pre = np.mean(np.array([results[indices[i]].x[:DParams]
                                 for i in range(topPer)]), axis=0)
    # average over D because only D in between bins is of importance
    DRes = np.array([(DRes_pre[i] + DRes_pre[i+1])/2
                     for i in range(DRes_pre.size-1)])
    # for fixed D_dermis
    if fixD_derm:
        DRes = np.concatenate((np.mean(np.array(
            [results[indices[i]].x[:DParams]
             for i in range(topPer)]), axis=0),
                               np.ones(1)*D_dermis))

    FRes_notNorm = np.mean(np.array([results[indices[i]].x[DParams:]
                                     for i in range(topPer)]), axis=0)
    FRes = FRes_notNorm - FRes_notNorm[0]
    # gathering standart deviation for top 1% of runs
    DSTD_pre = np.std(np.array([results[indices[i]].x[:DParams]
                                for i in range(topPer)]), axis=0)
    # average over D because only D in between bins is of importance
    DSTD = np.array([(DSTD_pre[i] + DSTD_pre[i+1])/2
                     for i in range(DSTD_pre.size-1)])
    # for fixed D_dermis
    if fixD_derm:
        DSTD_pre = np.concatenate((np.std(np.array(
            [results[indices[i]].x[:DParams]
             for i in range(topPer)]), axis=0),
                               np.zeros(1)))
        # average over D because only D in between bins is of importance
        DSTD = np.array([(DSTD_pre[i] + DSTD_pre[i+1])/2
                         for i in range(DSTD_pre.size-1)])

    FSTD = np.std(np.array([results[indices[i]].x[DParams:]
                            for i in range(topPer)]), axis=0)

    D_best_pre = results[indices[0]].x[:DParams]
    # average over D because only D in between bins is of importance
    D_best = np.array([(D_best_pre[i] + D_best_pre[i+1])/2
                       for i in range(D_best_pre.size-1)])

    # for fixed D_dermis
    if fixD_derm:
        D_best = np.concatenate((results[indices[0]].x[:DParams],
                                 np.ones(1)*D_dermis))

    F_best_notNorm = results[indices[0]].x[DParams:]
    F_best = F_best_notNorm - F_best_notNorm[0]

    # compute D and F and concentration profiles
    D, F = fp.computeDF(DRes, FRes, shape=segments)
    D_std, F_std = fp.computeDF(DSTD, FSTD, shape=segments)
    Db, Fb = fp.computeDF(D_best, F_best, shape=segments)
    # computing WMatrix and concentration profiles only for best run
    W = fp.WMatrixVar(Db, Fb, N, deltaXX)
    # computing concentration profiles
    ccRes = np.array([fp.calcC(cc[0], tt[j], W=W) for j in range(tt.size)])
    # -------------------------- loading results --------------------------- #

    # --------------------------- saving data ------------------------------- #
    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'concentrationRes.txt', ccRes, delimiter=',')
    # saving averaged DF
    np.savetxt(savePath+'DF.txt', np.array([D, D_std, F, F_std]).T,
               delimiter=',')
    # saving best DF
    np.savetxt(savePath+'DF_best.txt', np.array([Db, Fb]).T, delimiter=',')
    # saving Error of top 1% of runs
    np.savetxt(savePath+'minError.txt', Error[indices[:topPer]])
    # --------------------------- saving data ------------------------------- #

    # ------------------------- plotting data ------------------------------- #
    # plotting profiles
    # xx1 = deltaX[0]*np.arange(-9, 1)
    # xx2 = np.arange(1, N+1)*deltaX[1]
    # xx3 = np.arange(N+1, N+11)*deltaX[2]
    # xx = np.concatenate((xx1, xx2, xx3))
    if plot:
        xx = np.arange(102)
        plotConSkin(xx, cc, ccRes, tt, save=True, path=savePath)

        # plotting averaged D and F
        plt.figure(1)
        plt.errorbar(xx, D, c='r', marker='.', yerr=D_std)
        plt.xlabel('Bins')
        plt.ylabel('Diffusivity [µm$^2$/s]')
        plt.yscale('log')
        plt.savefig(savePath+'D.pdf')

        plt.figure(2)
        plt.errorbar(xx, F, c='r', marker='.',  yerr=F_std)
        plt.xlabel('Bins')
        plt.ylabel('Free energy [k$_B$T]')
        plt.savefig(savePath+'F.pdf')


def resFun(df, cc, tt, deltaX=1, debug=False, verb=False):
    '''
    cc and tt arrays with concentration profiles cc[:,i]
    for time tt[i] and tt[j] > tt[i] if j > i
    were cc is 2D array with cc.shape = (number of samples, number of bins)
    '''

    if len(cc.shape) == 1:
        # catches the case of differently sized concentration profiles
        # within cc and simply takes maximum as N
        N = np.max(np.array([cc[i].size for i in range(1, cc.size)]))
        M = cc.size
    else:
        M = cc[0, :].size  # number of concentration profiles
        N = cc[:, 0].size  # number of bins

    # pre-processing for d and f vectors
    dPre = df[:(N+2)]  # take input values from trust region algorithm
    # fPre = np.concatenate((np.zeros(1), df[(N+2):]))
    # completely free F
    fPre = df[(N+2):]
    # defined for keeping d and f constant in certain areas
    segments = np.concatenate((np.zeros(7), np.arange(1, N+1),
                               np.ones(15)*(N+1))).astype(int)
    d, f = fp.computeDF(dPre, fPre, shape=segments)
    # transition between discretization is within segment 1 at pos. 5
    # and segment 2 at pos. N+7-10
    # bin discretization widths are stored in deltaXX
    deltaXX = np.concatenate((np.ones(5)*deltaX[0],
                              np.ones(N+7)*deltaX[1],
                              np.ones(11)*deltaX[2]))
    W = fp.WMatrixVar(d, f, N, deltaXX, con=debug)
    T = al.expm(W)

    # check for detailed balance and conservation of concentration
    # (only for reflective boundaries)
    if debug:
        # numerical error min 100 times smaller than zero
        if abs(np.sum(np.sum(W, 0))) > 1E-2:
            print('Error: W Matrix is not row stochastic in rows: \n',
                  np.nonzero(abs(np.sum(W, 0)) > 1E-2), '\n')
            print('Row Sum:\n', np.sum(W, 0), '\n')
            print('Total Sum:\n', np.sum(np.sum(W, 0)), '\n')
            sys.exit()

        # same check for T matrix
        # if abs(np.sum(np.sum(T, 0))-T[:, 0].size) > T[:, 0].size*1E-2:
        #     print('Error: T Matrix is not row stochatic in rows: \n',
        #           np.nonzero(abs(np.sum(T, 0)-1) > 1E-2), '\n')
        #     print('Row Sum:\n', np.sum(T, 0), '\n')
        #     print('Total Sum:\n', np.sum(np.sum(T, 0)), '\n')
        #     sys.exit()

        deltaXC = np.concatenate((np.ones(4)*deltaX[0],
                                  np.ones(1)*(deltaX[0]+deltaX[1])/2,
                                  np.ones(N+6)*deltaX[1],
                                  np.ones(1)*(deltaX[1]+deltaX[2])/2,
                                  np.ones(10)*deltaX[2]))
        # deltaXC = deltaXX[:-1]
        con = np.sum(cc[0]*deltaXC)

        # compute profiles from c0 and do the same conservation check
        ccComp = np.array([fp.calcC(cc[0], t=tt[i], W=W)
                           for i in range(M)])

        if np.any(np.array([abs(np.sum(ccComp[i]*deltaXC)-con)
                            for i in range(M)]) > 0.01*con):
            print('Error: Computed concentration '
                  'is not conserved in profiles: \n',
                  np.nonzero(np.array([abs(np.sum(ccComp[i]*deltaXC)-con)
                                       for i in range(M)]) > 0.01*con))
            print([np.sum(ccComp[i]*deltaXC) for i in range(M)], '\n')
            print('concentration:\n', con)
            print('WMatrix Size:\n', W.shape)
            print('WMatrix Row Sum:\n', np.sum(W, 0))
            print('WMatrix 2Sum:\n', np.sum(np.sum(W, 0)))
            sys.exit()

    #  all residuals where c_0 profile is used for numerical computations
    Ri_0 = [cc[i] - fp.calcC(cc[0], tt[i], T=T)[7:(cc[i].size+7)]
            for i in range(1, M)]

    # residuals = np.concatenate((Ri_0[0], Ri_0[1], Ri_0[2]))
    # use roberts normalized residuals
    residuals = np.concatenate((600*Ri_0[0]/np.sqrt(73),
                                600*Ri_0[1]/np.sqrt(80),
                                600*Ri_0[2]/np.sqrt(80)))

    if (verb):
        # normalized version as used by Robert
        sigma = np.sqrt((np.sum((residuals[:73]**2)) +
                         np.sum((residuals[73:153]**2)) +
                         np.sum((residuals[153:233]**2))) / (M-1))
        E = 0.5*np.sum(residuals**2)  # non-normalized version (used by scipy)
        print("non-normalized: %f \nnormalized: %f \n" % (E, sigma))

    return residuals


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, bnds, cc, tt, deltaX=1, debug=False,
                 verb=False):

    optimize = ft.partial(resFun, cc=cc, tt=tt, deltaX=deltaX, debug=debug,
                          verb=verb)
    initVal = np.concatenate((DRange, FRange))  # starting values of ls-algo

    # running freeely with standart termination conditions
    # result = op.least_squares(optimize, initVal, bounds=bnds,
    #                           max_nfev=None, tr_solver='exact', verbose=2)

    for i in range(5):
        result = op.least_squares(optimize, initVal, bounds=bnds,
                                  max_nfev=50, verbose=2)
        initVal = result.x

    return result


def main():
    # ----------------- parsing command line inputs ------------------------- #
    parser = ap.ArgumentParser()
    parser.add_argument('-c', '--conservation', action='store_true',
                        help='turn on checks for conservation of '
                        'concentration')
    parser.add_argument('-p', dest='path', type=str,
                        help='define the relative path to data for analysis')
    args = parser.parse_args()
    # gathering path to data and setting verbosity and conservation mode
    path = args.path
    if args.conservation:
        conservation = True
    else:
        conservation = False
    # ---------------- parsing command line inputs ------------------------- #

    # --------------------------- reading profiles ------------------------- #
    # initial profile is generated by distributing concentration in gel phase
    # first 7 bins = gel, then 80 bins = epidermis, then 15 bins = deeper skin
    # np.concatenate((np.ones(10)*0.0025, np.zeros(90)))  # original profile
    cc = np.array([np.concatenate((np.ones(7)*0.0025, np.zeros(95))),
                   io.readData(path+'p10min.txt')[:73],
                   io.readData(path+'p100min.txt')[:80],
                   io.readData(path+'p1000min.txt')[:80]]).T

    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    # max number of measured points in epidermis
    N = max([cc[i].size for i in range(1, cc.size)])
    # computing discretization lengths
    # original way for discretization
    # X2 = 1  # discretization width in epidermis is 1µm
    # X1 = (400-(3.5*X2))/6.5  # transition between discretizations at bin 7
    # X3 = (20000-(3.5*X2))/6.5  # transition between discretizations at bin 93

    # using roberts discretization
    X2 = 1  # in epidermis 1µm
    X1 = (400-(2.5*X2))/4.5  # 400µm in epidermis
    X3 = (20000-(4.5*X2))/10.5  # 2cm in deeper skin layers

    deltaX = np.array([X1, X2, X3])
    # --------------------------- reading profiles -------------------------- #

    # -------------------- starting ls-optimization ------------------------- #
    # setting bounds, D first and F second
    # there are a total of 2N+4 fit parameters,
    # N+2 for D (N in epidermis and one for gel and dermis)
    # and N+2 for F, now freely fitted
    bndsDUpper = np.ones(N+2)*1000
    bndsFUpper = np.ones(N+2)*60
    # bndsDLower = np.zeros(N+2)
    bndsDLower = np.zeros(N+2)
    bndsFLower = np.ones(N+2)*40
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    # setting initial conditions
    # D is randomly chosen at each point and F is constant throughout
    # DInit = np.random.rand(N+2, 10)
    RUNS = 1000
    DInit = np.concatenate((np.random.rand(15, RUNS),
                            (np.random.rand(67, RUNS)*700+100)))
    FInit = 50

    # trying Roberts results as initial data
    # path = ('/Users/AmanuelWK/Dropbox/PhD/Projects/FokkerPlanckModeling/'
    #         'Skin/Data/RobertsResults/')
    # d = np.loadtxt(path+'D.txt')
    # f = np.loadtxt(path+'F.txt')
    # DInit = np.concatenate((np.ones(1)*d[0], d[7:-15], np.ones(1)*d[-1]))
    # # taking care of negative values in roberts D-vector
    # # for i in range(DInit.size):
    # #     if DInit[i] < 0:
    # #         DInit[i] = 0
    # FInit = np.concatenate((f[7:-15], np.ones(1)*f[-1]))

    results = []
    for i in range(RUNS):
        results.append(optimization(DRange=DInit[:, i],
                                    FRange=FInit*np.ones(N+2), bnds=bnds,
                                    cc=cc, tt=tt, debug=conservation,
                                    verb=False, deltaX=deltaX))
        np.save('result.npy', np.array(results))
    # -------------------- starting ls-optimization ------------------------- #

    # doing analysis and plotting data
    xx = np.arange(102)
    analysis(np.array(results), xx=xx, cc=cc, tt=tt, deltaX=deltaX, plot=True,
             per=0.1, savePath=None, fixD_derm=False, D_dermis=None)


if __name__ == "__main__":
    startTime = time.time()
    main()
    print("Execution time is: %f minutes" % ((time.time() - startTime)/60))
