import numpy as np
import inputOutput as io
import FPModel as fp
import argparse as ap
import matplotlib.pyplot as plt
import os
import sys

'''
this script saves D, F and computed and experimental concentration profiles
 for skin analysis script
'''


# for printing c-profiles
def plotConSkin(xx, cc, ccRes, tt, save=False, path=None, deltaXX=None):

    M = cc.size  # number of profiles
    N = ccRes[0, :].size  # number of bins
    if deltaXX is None:
        deltaXX = np.ones(N+1)
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
    cc = np.array([np.concatenate((np.ones(7)*0.0025, np.zeros(95))),
                   io.readData(path+'p10min.txt')[:73],
                   io.readData(path+'p100min.txt')[:80],
                   io.readData(path+'p1000min.txt')[:80]]).T

    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    # max number of measured points in epidermis
    N = max([cc[i].size for i in range(1, cc.size)])
    M = cc.size
    # computing discretization lengths
    # # my original discretization
    # X2 = 1  # discretization length in epidermis is 1µm
    # X1 = (400-(3.5*X2))/6.5  # transition between discretizations at bin 7
    # X3 = (20000-(3.5*X2))/6.5  # transition between discretizations at bin 83
    # using roberts discretization
    X2 = 1  # in epidermis 1µm
    X1 = (400-(2.5*X2))/4.5  # 400µm in epidermins
    X3 = (20000-(4.5*X2))/10.5  # 2cm in deeper skin layers
    deltaX = np.array([X1, X2, X3])
    # vector of discretizations
    deltaXX = np.concatenate((np.ones(5)*deltaX[0],
                              np.ones(N+7)*deltaX[1],
                              np.ones(11)*deltaX[2]))
    # vector of different segments
    segments = np.concatenate((np.zeros(7), np.arange(1, N+1),
                               np.ones(15)*(N+1))).astype(int)
    # ------------------- experimental parameters ----------------------- #

    # -------------------------- loading results --------------------------- #
    results = np.load(path+'result.npy')

    I = results.size  # number of different initial conditions
    topPer = np.ceil(0.1*I).astype(int)  # number for top 1% of the runs

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
    # for fixed DSol
    # DParams = N+1
    # D_dermis = 67.341  # value from PNAS paper
    DParams = N+2

    # gathering mean of F and D for best 1% of runs
    DRes = np.mean(np.array([results[indices[i]].x[:DParams]
                             for i in range(topPer)]), axis=0)
    # for fixed D_dermis
    # DRes = np.concatenate((np.mean(np.array([results[indices[i]].x[:DParams]
                                            #  for i in range(topPer)]), axis=0),
                        #    np.ones(1)*D_dermis))

    FRes_notNorm = np.mean(np.array([results[indices[i]].x[DParams:]
                                     for i in range(topPer)]), axis=0)
    FRes = FRes_notNorm - FRes_notNorm[0]
    # gathering standart deviation for top 1% of runs
    DSTD = np.std(np.array([results[indices[i]].x[:DParams]
                            for i in range(topPer)]), axis=0)
    # for fixed D_dermis
    # DSTD = np.concatenate((np.std(np.array([results[indices[i]].x[:DParams]
                                            # for i in range(topPer)]), axis=0),
                        #    np.zeros(1)))

    FSTD = np.std(np.array([results[indices[i]].x[DParams:]
                            for i in range(topPer)]), axis=0)

    D_best = results[indices[0]].x[:DParams]
    # for fixed D_dermis
    # D_best = np.concatenate((results[indices[0]].x[:DParams],
                            #  np.ones(1)*D_dermis))
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
    xx = np.arange(102)
    plotConSkin(xx, cc, ccRes, tt, save=True, path=savePath)

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
