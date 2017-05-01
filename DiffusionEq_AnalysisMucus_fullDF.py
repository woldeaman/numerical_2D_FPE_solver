import numpy as np
import inputOutput as io
import FPModel as fp
import argparse as ap
import matplotlib.pyplot as plt
import os
import sys

'''
this script saves D, F and computed and experimental concentration profiles
 for full DF mucus analysis script
'''


# for printing analytical solution
def plotCon(xx, cc, ccRes, tt, save=False, path=None):

    M = cc[:, 0].size  # number of profiles
    N = cc[0, :].size  # number of bins
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    # plotting concentration profiles
    l1s = []  # for sperate legends
    l2s = []
    colors = ['r', 'm', 'c', 'b', 'y', 'k', 'g']

    plt.figure(0)
    for j in range(1, M):
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.xlabel('Distance [µm]')
        plt.ylabel('Concentration [µM]')
        l1, = plt.plot(xx, cc[j, :], '--', color=colors[j])

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


def main():
    # --------------- parsing command line inputs --------------------------- #
    parser = ap.ArgumentParser()
    parser.add_argument('-p', dest='path', type=str,
                        help='define the path to data for analysis')
    args = parser.parse_args()
    # gathering path to data and name of experiment
    path = args.path
    savePath = path+'/results/'

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    # ------------------- experimental parameters ----------------------- #
    # reading profiles and take only samples for 4 different time points
    data = io.readData(path+'Ch3_Negative.csv', sep=',')  # change seperator according to format
    xx = data[:, 0]
    cc = np.array([data[:, 1], data[:, 31], data[:, 61], data[:, 91]]).T

    # pre processing of profiles
    xx, cc = io.preProcessing(xx, cc)  # smoothing and discarding negative c
    N = cc[:, 0].size  # number of discretization bins
    M = cc[0, :].size  # number of profiles
    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 4  # concentration of peptide solution in µM
    # ------------------- experimental parameters ----------------------- #

    # -------------------------- loading results --------------------------- #
    results = np.load(path+'result.npy')

    I = results.size  # number of different initial conditions
    topPer = np.ceil(0.01*I).astype(int)  # number for top 1% of the runs

    # gathering data from simulations
    # loading error values, factor two, because of cost function definition
    Error = np.array([np.sqrt(np.sum(result[i].fun / ((M-1)*N)))
                      for i in range(I)])
    indices = np.argsort(Error)  # for sorting according to error

    # gathering mean of F and D for best 1% of runs
    DRes = np.mean(np.array([results[indices[i]].x[:dim+1]
                             for i in range(topPer)]), axis=0)
    FRes = np.mean([np.concatenate((np.zeros(1),
                                    results[indices[i]].x[dim+1:]))
                    for i in range(topPer)], axis=0)
    # gathering standart deviation for top 1% of runs
    DSTD = np.std(np.array([results[indices[i]].x[:dim+1]
                            for i in range(topPer)]), axis=0)
    FSTD = np.std([np.concatenate((np.zeros(1), results[indices[i]].x[dim+1:]))
                   for i in range(topPer)], axis=0)
    # setting D and F
    D = DRes
    F = FRes

    # computing WMatrix
    W = fp.WMatrix(D, F, bc='open1side')
    # computing concentration profiles
    ccRes = np.array([fp.calcC(cc[0], tt[j], W=W) for j in range(tt.size)])
    # -------------------------- loading results --------------------------- #

    # --------------------------- saving data ------------------------------- #
    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'concentrationRes.txt', ccRes, delimiter=',')
    # saving averaged DF
    np.savetxt(savePath+'DF.txt', np.array([D, D_std, F, F_std]).T,
               delimiter=',')
    # saving Error of top 1% of runs
    np.savetxt(savePath+'minError.txt', Error[indices[:topPer]])
    # --------------------------- saving data ------------------------------- #

    # ------------------------- plotting data ------------------------------- #
    # plotting profiles
    xx = np.arange(N)
    plotCon(xx, cc, ccRes, tt, save=True, path=savePath)

    # plotting averaged D and F
    plt.figure(1)
    plt.errorbar(xx, D, c='r', marker='.', yerr=D_std)
    plt.xlabel('Distance [µm]')
    plt.ylabel('Diffusivity [µm$^2$/s]')
    plt.yscale('log')
    plt.savefig(savePath+'D.pdf')

    plt.figure(2)
    plt.errorbar(xx, F, c='r', marker='.',  yerr=F_std)
    plt.xlabel('Distance [µm]')
    plt.ylabel('Free energy [k$_B$T]')
    plt.savefig(savePath+'F.pdf')


if __name__ == "__main__":
    main()
