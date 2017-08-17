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
    D_pre = np.mean(np.array([result[indices[i]].x[:2]
                              for i in range(nbr)]), axis=0)
    F_preNoGauge = np.mean(np.array([result[indices[i]].x[2:4]
                                     for i in range(nbr)]), axis=0)
    F_pre = F_preNoGauge - F_preNoGauge[0]
    sigParamsDF_mean = np.mean(np.array([result[indices[i]].x[4:]
                                         for i in range(nbr)]), axis=0)
    xx_ext = np.concatenate((-deltaX*np.ones(1), xx))
    D_mean = np.array([fp.sigmoidalDF(D_pre, sigParamsDF_mean[0],
                                      sigParamsDF_mean[1], x) for x in xx_ext])
    F_mean = np.array([fp.sigmoidalDF(F_pre, sigParamsDF_mean[2],
                                      sigParamsDF_mean[3], x) for x in xx_ext])

    # gathering standart deviation for top 1% of runs
    # computed from gauß errorpropagation for errors of D1, D2 or F1, F2
    # contribution of t_D, d_D, t_F and d_F neglected for now
    DSTD_pre = np.std(np.array([result[indices[i]].x[:2]
                                for i in range(nbr)]), axis=0)
    FSTD_pre = np.std(np.array([result[indices[i]].x[2:4]
                                for i in range(nbr)]), axis=0)
    DSTD = np.array([np.sqrt(
        ((0.5 - sp.erf(
            (x-sigParamsDF_mean[0])/(np.sqrt(2)*sigParamsDF_mean[1]))/2) *
         DSTD_pre[0])**2 +
        ((0.5 + sp.erf(
            (x-sigParamsDF_mean[0])/(np.sqrt(2)*sigParamsDF_mean[1]))/2) *
         DSTD_pre[1])**2) for x in xx_ext])
    FSTD = np.array([np.sqrt(
        ((0.5 - sp.erf(
            (x-sigParamsDF_mean[2])/(np.sqrt(2)*sigParamsDF_mean[3]))/2) *
         FSTD_pre[0])**2 +
        ((0.5 + sp.erf(
            (x-sigParamsDF_mean[2])/(np.sqrt(2)*sigParamsDF_mean[3]))/2) *
         FSTD_pre[1])**2) for x in xx_ext])
    sigParamsDF_STD = np.std(np.array([result[indices[i]].x[4:]
                                       for i in range(nbr)]), axis=0)

    # gathering best D and F for computation of profiles
    D_best_pre = result[indices[0]].x[:2]
    F_best_preNoGauge = result[indices[0]].x[2:4]
    F_best_pre = F_best_preNoGauge - F_best_preNoGauge[0]
    sigParamsDF_best = result[indices[0]].x[4:]
    D_best = np.array([fp.sigmoidalDF(D_best_pre, sigParamsDF_best[0],
                                      sigParamsDF_best[1], x) for x in xx_ext])
    F_best = np.array([fp.sigmoidalDF(F_best_pre, sigParamsDF_best[2],
                                      sigParamsDF_best[3], x) for x in xx_ext])

    # computing WMatrix for open boundary conditions
    W, W10 = fp.WMatrixGrima(D_best, F_best, deltaX=deltaX, bc='open1side')
    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[:, 0], tt[j], W=W, W10=W10, c0=c0,
                               bc='open1side') for j in range(tt.size)]).T
    # -------------------------- loading results --------------------------- #

    # --------------------------- saving data ------------------------------- #
    # extended xx and cc vector for boundary condition
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
    np.savetxt(savePath+'DF.txt', np.c_[xx_ext, D_mean, DSTD, F_mean, FSTD],
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

    # saving data to excel spreadsheet
    workbook = xl.Workbook(savePath+'results.xlsx')
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    # writing headers
    worksheet.write('A1', 'D_sol [µm^2/s]', bold)
    worksheet.write('B1', 'D_muc [µm^2/s]', bold)
    worksheet.write('C1', 'F_muc [kT]', bold)
    worksheet.write('D1', 'layer t_D [µm]', bold)
    worksheet.write('E1', 'layer t_F [µm]', bold)
    worksheet.write('F1', 'layer d_D [µm]', bold)
    worksheet.write('G1', 'layer d_F [µm]', bold)
    worksheet.write('H1', 'min E [+/- µM]', bold)
    # writing entries
    worksheet.write('A2', '%.2f +/- %.2f' % (D_best[0], DSTD[0]))
    worksheet.write('B2', '%.2f +/- %.2f' % (D_best[-1], DSTD[-1]))
    worksheet.write('C2', '%.2f +/- %.2f' % (F_best[-1], FSTD[-1]))
    worksheet.write('D2', '%.2f +/- %.2f' % (sigParamsDF_best[0],
                                             sigParamsDF_STD[0]))
    worksheet.write('E2', '%.2f +/- %.2f' % (sigParamsDF_best[2],
                                             sigParamsDF_STD[2]))
    worksheet.write('F2', '%.2f +/- %.2f' % (sigParamsDF_best[1],
                                             sigParamsDF_STD[1]))
    worksheet.write('G2', '%.2f +/- %.2f' % (sigParamsDF_best[3],
                                             sigParamsDF_STD[3]))
    worksheet.write('H2', '%.2f' % Error[indices[0]])

    # adjusting cell widths
    worksheet.set_column(0, 5, len('layer d_D, d_F [µm]'))
    workbook.close()
    # --------------------------- saving data ------------------------------- #

    # ------------------------- plotting data ------------------------------- #
    if plot:
        # plotting profiles
        ps.plotCon(xx, cc, ccRes, tt, c0=c0, save=True, path=savePath)
        # plotting averaged D and F
        ps.plotDF(xx, D_mean, F_mean, style='-.', D_STD=DSTD, F_STD=FSTD,
                  save=True, path=savePath, scale='linear')


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, xx, tt, deltaX=1, c0=None, verb=False):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include: discretization width: deltaX,
    distance of transition regime: dist and concentration at left boundary: c0
    --> vector df now includes parameters for sigmoidal DF profiles
    df = [d1, d2, f1, f2, t_D, d_D, t_F, d_F]
    '''

    M = cc[0, :].size  # number of concentration profiles
    N = cc[:, 0].size  # number of bins

    # N paramters to be optimized D1, D2, F1, F2, t_D, d_D, t_F, d_F
    d = df[:2]
    f = df[2:4]  # letting F completely free
    # computing sigmoidal d and f profiles
    t_D, d_D, t_F, d_F = df[4], df[5], df[6], df[7]
    xx_ext = np.concatenate((-np.ones(1)*deltaX, xx))  # extended x for c0 bin
    D = np.array([fp.sigmoidalDF(d, t_D, d_D, x) for x in xx_ext])
    F = np.array([fp.sigmoidalDF(f, t_F, d_F, x) for x in xx_ext])
    # calculating W and T matrix and extra variables for open BCs
    W, W10 = fp.WMatrixGrima(D, F, deltaX, bc='open1side')
    # catching singular matrix exception
    try:
        Q = la.inv(W)  # inverse of W
    except la.linalg.LinAlgError:
        print('Values for which singular Matrix occured: \n')
        print('D: \n', d, '\nF: \n', f, '\nt_D, d_D: \n', [t_D, d_D],
              '\nt_F, d_F: \n', [t_F, d_F], '\n', )
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

    # print('Current iteration:\n')
    # print('D: \n', d, '\nF: \n', f, '\nt_D, d_D: \n', [t_D, d_D],
    #       '\nt_F, d_F: \n', [t_F, d_F], '\n', )
    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, tdRange, bnds, cc, xx, tt, deltaX=1, c0=None,
                 verb=0):

    if verb == -1:
        funcVerb = True
        scpVerb = 0
    else:
        funcVerb = False
        scpVerb = verb

    optimize = ft.partial(resFun, cc=cc, xx=xx, tt=tt, deltaX=deltaX, c0=c0,
                          verb=funcVerb)

    initVal = np.concatenate((DRange, FRange, tdRange))
    # running freely with standart termination conditions
    result = op.least_squares(optimize, initVal, bounds=bnds,
                              max_nfev=None, verbose=scpVerb)

    return result


def main():
    # reading input and setting up analysis
    (verbosity, Runs, ana, deltaX, c0, xx, cc, tt,
     bnds, FInit, DInit) = io.startUp('fullDF')

    # ---------------- option for analysis only --------------------------- #
    if ana:
        print('\nDoing analysis only.')
        res = np.load('result.npy')
        analysis(np.array(res), c0=c0, xx=xx, cc=cc, tt=tt, plot=True, per=0.1)
        print('\nPlots have been made and data was extraced and saved.')
        sys.exit()
    # ---------------- option for analysis only --------------------------- #

    # setting reasonable bounds for F and D
    DBound = 1000
    FBound = 20

    bndsDUpper = np.ones(2)*DBound
    bndsFUpper = np.ones(2)*FBound
    # otherwise singular matrix due to vanishing D --> vanishing rates
    bndsDLower = np.ones(2)*1E-5
    bndsFLower = np.ones(2)*(-FBound)
    # bounds for interface position and layer thickness zero and max x position
    # using independent parameters for F and D profiles --> 4 params
    tdBoundsLower = np.zeros(4)
    tdBoundsUpper = np.ones(4)*np.max(xx)
    bnds = (np.concatenate((bndsDLower, bndsFLower, tdBoundsLower)),
            np.concatenate((bndsDUpper, bndsFUpper, tdBoundsUpper)))
    FInit = 0
    DInit = (np.random.rand(2, Runs)*DBound)
    # order is [t_D, d_D, t_F, d_F]
    tdInit = np.array([np.max(xx)/4, deltaX]*2)

    # TODO: try normalized parameters --> p_i e [0, 1]

    results = []
    for i in range(Runs):
        results.append(optimization(DRange=DInit[:, i],
                                    FRange=FInit*np.ones(2), tdRange=tdInit,
                                    bnds=bnds, cc=cc, xx=xx, tt=tt,
                                    deltaX=deltaX, c0=c0, verb=verbosity))
        np.save('result.npy', np.array(results))

    analysis(np.array(results), c0=c0, xx=xx, cc=cc, tt=tt, plot=True, per=0.1)

    # returns number of runs in order to compute average time per run
    return Runs


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %.2f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
