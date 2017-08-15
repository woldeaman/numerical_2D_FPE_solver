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
import xlsxwriter as xl
import os
import sys

startTime = time.time()


def analysis(result, c0, transIdx, dist, xx=None, cc=None, tt=None, plot=False,
             per=0.1, savePath=None):
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

    K = result.size  # number of different transition sizes
    I = result[0].size  # number of different initial conditions
    nbr = np.ceil(per*I).astype(int)  # number for top x% of the runs
    if plot:
        M = tt.size  # number of different profiles

    N = cc[:, 0].size  # number of bins, -1 because c0 const.
    n = int(sp.binom(M, 2))  # number of combinations for different c-profiles
    if xx is None:
        xx = np.arange(N)
    deltaX = abs(xx[0] - xx[1])
    distanceMuM = (dist-1)*deltaX
    # ----------------- setting working parameters --------------------- #

    # -------------------------- loading results --------------------------- #
    # gathering data from simulations
    # loading error values, factor two, because of cost function definition
    Error = np.array([[np.sqrt(result[k][i].cost*2/(N*n)) for i in range(I)]
                      for k in range(K)])
    indices = np.argsort(Error)  # for sorting according to error

    # minimal error for each transition layer size
    EMin = np.array([np.min(Error[k, :]) for k in range(K)])
    ESTD = np.array([np.std(Error[k, :]) for k in range(K)])
    indexLayer = np.argmin(EMin)

    # gathering mean of F and D for best x% of runs
    D = np.mean(np.array([result[indexLayer][indices[indexLayer, i]].x[:2]
                          for i in range(nbr)]), axis=0)
    # gauging F to be zero at the inlet
    F_pre = np.mean(np.array([result[indexLayer][indices[indexLayer, i]].x[2:]
                              for i in range(I)]), axis=0)
    F = F_pre - F_pre[0]

    # gathering standart deviation for top x% of runs
    DSTD = np.std(np.array([result[indexLayer][indices[indexLayer, i]].x[:2]
                           for i in range(nbr)]), axis=0)
    FSTD = np.std(np.array([result[indexLayer][indices[indexLayer, i]].x[2:]
                            for i in range(nbr)]), axis=0)

    # transforming into full profile
    bestDist = dist[indexLayer]  # found optimal transition layer thickness
    segments = np.concatenate((np.ones(transIdx+1)*0,
                               np.ones(N-transIdx)*1)).astype(int)
    d_mean, f_mean = fp.computeDF(D, F, shape=segments, mode='transition',
                                  transiBin=transIdx, dx=bestDist)
    d_STD, f_STD = fp.computeDF(DSTD, FSTD, shape=segments, mode='transition',
                                transiBin=transIdx, dx=bestDist)

    # gathering best D and F for computation of profiles
    D_best = result[indexLayer][indices[indexLayer, 0]].x[:2]
    F_best_pre = result[indexLayer][indices[indexLayer, 0]].x[2:]
    F_best = F_best_pre - F_best_pre[0]

    # computing WMatrix for open boundary conditions
    d_best, f_best = fp.computeDF(D_best, F_best, shape=segments,
                                  mode='transition', transiBin=transIdx,
                                  dx=bestDist)
    W, W10 = fp.WMatrix(d_best, f_best, deltaX=deltaX, bc='open1side')

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
    np.savetxt(savePath+'DF.txt', np.c_[xx_ext, d_mean, d_STD, f_mean, f_STD],
               delimiter=',',
               header=('Diffusivity and free energy profiles from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: average diffusivity [micro_m^2/s]\n'
                       'cloumn3: stdev of diffusivity [+/- micro_m^2/s]\n'
                       'cloumn4: average free energy [k_BT]\n'
                       'cloumn5: stdev of free energy [+/- k_BT]\n'))
    # saving best DF
    np.savetxt(savePath+'DF_best.txt', np.c_[xx_ext, d_best, f_best],
               delimiter=',',
               header=('Diffusivity and free energy profiles with lowest '
                       'error from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: diffusivity [micro_m^2/s]\n'
                       'cloumn3: free energy [k_BT]\n'))

    # saving Error of top 1% of runs
    np.savetxt(savePath+'minError.txt', Error[indexLayer, :nbr], delimiter=',',
               header=('Minimal error for top %.2f %% runs.' % (per*100)))

    # saving data to excel spreadsheet
    workbook = xl.Workbook(savePath+'results.xlsx')
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    # writing headers
    worksheet.write('A1', 'D_sol [µm^2/s]', bold)
    worksheet.write('B1', 'D_muc [µm^2/s]', bold)
    worksheet.write('C1', 'F_muc [kT]', bold)
    worksheet.write('D1', 'layer d [µm]', bold)
    worksheet.write('E1', 'min E [+/- µM]', bold)
    # writing entries
    worksheet.write('A2', '%.2f +/- %.2f' % (D_best[0], DSTD[0]))
    worksheet.write('B2', '%.2f +/- %.2f' % (D_best[1], DSTD[1]))
    worksheet.write('C2', '%.2f +/- %.2f' % (F_best[1], FSTD[1]))
    worksheet.write('D2', '%.2f' % distanceMuM[indexLayer])
    worksheet.write('E2', '%.2f' % EMin[indexLayer])
    # adjusting cell widths
    worksheet.set_column(0, 5, len('minError [+/- µM]'))
    workbook.close()
    # --------------------------- saving data ------------------------------- #

    # ------------------------- plotting data ------------------------------- #
    if plot:
        ps.plotMinError(distanceMuM, EMin, ESTD, save=True,
                        path=savePath)
        # for plotting best D and F in the same figure
        ps.plotDF(xx, d_best, f_best, style='-', save=True, path=savePath)
        # plotting concentration profiles for best run
        ps.plotConTrans(xx, cc, ccRes, c0, tt, transIdx,
                        layerD=distanceMuM[indexLayer], save=True,
                        path=savePath)


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, tt, dist, transIdx, deltaX=1, c0=None, verb=False):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include: discretization width: deltaX,
    distance of transition regime: dist and concentration at left boundary: c0
    '''

    M = cc[0, :].size  # number of concentration profiles
    N = cc[:, 0].size  # number of bins

    # 3 paramters to be optimized, D', D and F
    dPre = df[:2]
    fPre = df[2:]  # letting F completely free
    # total of N+1 bins because of constant c0 BCs, WMatrix for this case
    # discards first row and thus additional bin is needed, but same D and F
    segments = np.concatenate((np.ones(transIdx+1)*0,
                               np.ones(N-transIdx)*1)).astype(int)
    d, f = fp.computeDF(dPre, fPre, shape=segments,
                        mode='transition', transiBin=transIdx, dx=dist)

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
def optimization(DRange, FRange, bnds, cc, tt, Dist, transIdx, deltaX=1,
                 c0=None, verb=0):

    if verb == -1:
        funcVerb = True
        scpVerb = 0
    else:
        funcVerb = False
        scpVerb = verb

    optimize = ft.partial(resFun, cc=cc, tt=tt, dist=Dist, transIdx=transIdx,
                          deltaX=deltaX, c0=c0, verb=funcVerb)

    initVal = np.concatenate((DRange, FRange))
    # running freely with standart termination conditions
    result = op.least_squares(optimize, initVal, bounds=bnds,
                              max_nfev=None, verbose=scpVerb)

    return result


def main():
    # reading input and setting up analysis
    (verbosity, Runs, ana, deltaX, c0, xInter, xx, cc, tt, bnds, FInit, DInit,
     distances, TransIndex) = io.startUp('twoBox')

    # ---------------- option for analysis only --------------------------- #
    if ana:
        print('\nDoing analysis only.')
        res = np.load('result.npy')
        analysis(np.array(res), transIdx=TransIndex, dist=distances, c0=c0,
                 xx=xx, cc=cc, tt=tt, plot=True, per=0.1)
        print('\nPlots have been made and data was extraced and saved.')
        sys.exit()
    # ---------------- option for analysis only --------------------------- #

    results = []
    for d in distances:
        for i in range(Runs):
            temp = []
            if verbosity > 0:
                print('\nNow at transition layer thicknes %s µm'
                      % ((d-1)*deltaX))
            temp.append(optimization(DRange=DInit[:, i],
                                     FRange=np.ones(2)*FInit, Dist=d,
                                     transIdx=TransIndex,
                                     bnds=bnds, cc=cc, tt=tt, deltaX=deltaX,
                                     c0=c0, verb=verbosity))
            np.save('temp.npy', np.array(temp))
        results.append(np.array(temp))
        np.save('result.npy', np.array(results))

    analysis(np.array(results), transIdx=TransIndex, dist=distances, c0=c0,
             xx=xx, cc=cc, tt=tt, plot=True, per=0.1)

    # returns number of runs in order to compute average time per run
    return Runs, distances.size


if __name__ == "__main__":
    runs, dist = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %.2f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs*dist)))
