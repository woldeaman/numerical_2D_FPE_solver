import numpy as np
import inputOutput as io
import FPModel as fp
import argparse as ap
import scipy.special as sp
import plottingScripts as ps
import os

'''
this script saves D, F and computed and experimental concentration profiles
 for full DF mucus analysis script
'''


def main():
    # --------------- parsing command line inputs --------------------------- #
    parser = ap.ArgumentParser()
    parser.add_argument('-p', dest='path', type=str,
                        help='define the path to data for analysis')
    parser.add_argument('-n', dest='name', type=str,
                        help='define the name of the file for the analyzed'
                        ' c-profiles')
    parser.add_argument('-t', dest='top', type=float,
                        help='set the percentage of best runs to average over,'
                        ' default is 0.1 - top 10%%')
    args = parser.parse_args()
    # gathering path to data and name of experiment
    if args.top is None:
        topPer = 0.1
    else:
        topPer = args.top
    path = args.path
    name = args.name
    savePath = path+'/results/'

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    # --------------- parsing command line inputs --------------------------- #

    # ------------------- experimental parameters ----------------------- #
    # change seperator for reading profiles according to format
    data = io.readData(path+name, sep=',')
    xx, cc = data[:, 0], data[:, 1:]

    N = cc[:, 0].size  # number of discretization bins
    M = cc[0, :].size  # number of profiles
    n = int(sp.binom(M, 2))  # number of combinations for different c-profiles
    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 4  # concentration of peptide solution in µM
    # ------------------- experimental parameters ----------------------- #

    # -------------------------- loading results --------------------------- #
    results = np.load(path+'result.npy')  # this might take a while

    I = results.size  # number of different initial conditions
    nbr = np.ceil(topPer*I).astype(int)  # number for top 1% of the runs

    # gathering data from simulations
    # loading error values, factor two, because of cost function definition
    Error = np.array([np.sqrt(np.sum(results[i].fun**2 / (n*N)))
                      for i in range(I)])
    indices = np.argsort(Error)  # for sorting according to error

    # gathering mean of F and D for best 1% of runs
    D = np.mean(np.array([results[indices[i]].x[:N+1]
                          for i in range(nbr)]), axis=0)
    F_pre = np.mean(np.array([results[indices[i]].x[N+1:]
                              for i in range(nbr)]), axis=0)
    F = F_pre - F_pre[0]
    # gathering standart deviation for top 1% of runs
    DSTD = np.std(np.array([results[indices[i]].x[:N+1]
                            for i in range(nbr)]), axis=0)
    FSTD = np.std(np.array([results[indices[i]].x[N+1:]
                            for i in range(nbr)]), axis=0)

    # gathering best D and F for computation of profiles
    D_best = results[indices[0]].x[:N+1]
    F_best_pre = results[indices[0]].x[N+1:]
    F_best = F_best_pre - F_best_pre[0]

    # computing WMatrix for open boundary conditions
    W, W10 = fp.WMatrix(D_best, F_best, deltaX=deltaX, bc='open1side')

    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[:, 0], tt[j], W=W, W10=W10, c0=c0,
                               bc='open1side') for j in range(tt.size)]).T
    # -------------------------- loading results --------------------------- #

    # --------------------------- saving data ------------------------------- #
    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'concentrationRes.txt', np.c_[xx, ccRes],
               delimiter=',',
               header=('Numerically computed concentration profiles\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: c-profile [micro_M] for t_0 = %i min\n'
                       'cloumn3: c-profile [micro_M] for t_1 = %i min\n'
                       'cloumn4: c-profile [micro_M] for t_2 = %i min\n'
                       'cloumn5: c-profile [micro_M] for t_3 = %i min\n' %
                        (tt[0]/60, tt[1]/60, tt[2]/60, tt[3]/60)))
    # saving averaged DF
    xx_DF = np.concatenate((-deltaX*np.ones(1), xx))
    np.savetxt(savePath+'DF.txt', np.c_[xx_DF, D, DSTD, F, FSTD],
               delimiter=',',
               header=('Diffusivity and free energy profiles from analysis\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: average diffusivity [micro_m^2/s]\n'
                       'cloumn3: stdev of diffusivity [+/- micro_m^2/s]\n'
                       'cloumn4: average free energy [k_BT]\n'
                       'cloumn5: stdev of free energy [+/- k_BT]\n'))

    # saving Error of top 1% of runs
    np.savetxt(savePath+'minError.txt', Error[indices[:nbr]], delimiter=',',
               header=('Minimal error for top %.2f %% runs.' % (topPer*100)))
    # --------------------------- saving data ------------------------------- #

    # ------------------------- plotting data ------------------------------- #
    # plotting profiles
    ps.plotCon(xx, cc, ccRes, tt, save=True, path=savePath)

    # plotting averaged D and F
    ps.plotDF(xx, D, F, D_STD=DSTD, F_STD=FSTD, save=True, path=savePath)


if __name__ == "__main__":
    main()
