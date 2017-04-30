import numpy as np
import inputOutput as io
import FPModel as fp
import scipy.special as sp
import argparse as ap
import os
import sys
import xlsxwriter as xl
# import os
# for debugging
import sys

'''
this script saves D, F and computed and experimental concentration profiles
 for skin analysis script
'''


def main():
    # --------------- parsing command line inputs --------------------------- #
    parser = ap.ArgumentParser()
    parser.add_argument('-p', dest='path', type=str,
                        help='define the path to data for analysis')
    parser.add_argument('name', help='defines name of experimental'
                        ' concentration data for which analyis was performed')
    parser.add_argument('-d', '--dsol_const', action='store_true', help='sets '
                        'flag for Dsol = const. mode')
    args = parser.parse_args()
    # gathering path to data and name of experiment
    path = args.path
    name = args.name
    # folder to save figures in changes depending on system
    if sys.platform == "darwin":  # folder for linux
        savePath = '/Users/AmanuelWK/Desktop/%s/Data/' % name
    elif sys.platform.startswith("linux"):  # folder for mac
        savePath = '/home/amanuelwk/Desktop/%s/Data/' % name

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    '''Add option for active input '''

    # ------------------- experimental parameters ----------------------- #
    cData = np.array([np.concatenate((np.ones(10)*0.0025, np.zeros(90))),
                      io.readData(path+'p10min.txt')[:73],
                      io.readData(path+'p100min.txt')[:80],
                      io.readData(path+'p1000min.txt')[:80]]).T

    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    # max number of measured points in epidermis
    N = max([cc[i].size for i in range(1, cc.size)])
    # computing discretization lengths
    X2 = 1  # discretization length in epidermis is 1µm
    X1 = (400-(3.5*X2))/6.5  # transition between discretizations at bin 7
    X3 = (20000-(3.5*X2))/6.5  # transition between discretizations at bin 83
    deltaX = np.array([X1, X2, X3])
    # vector of discretizations
    deltaXX = np.concatenate((np.ones(7)*deltaX[0],
                              np.ones(N+6)*deltaX[1],
                              np.ones(8)*deltaX[2]))
    # vector of different segments
    segments = np.concatenate((np.ones(10)*0, np.arange(1, N+1),
                               np.ones(10)*(N+1))).astype(int)
    # -------------------------- loading results --------------------------- #
    results = np.load(path+'result.npy')

    I = results.size  # number of different initial conditions
    topPer = 0.01*I  # number for top 1% of the runs

    # gathering data from simulations
    # loading error values, factor two, because of cost function definition
    Error = np.array([600*np.sqrt(np.sum((results[i].cost[:73]**2)/73) +
                      np.sum((results[i].cost[73:153]**2)/80) +
                      np.sum((results[i].cost[153:]**2)/80) / (M-1))
                      for i in range(I)])
    indices = np.argsort(Error)  # for sorting according to error

    # gathering F and D for best 1% of runs
    DRes = np.mean(np.array([results[indices[i]].x[:82]
                             for i in range(topPer)]), axis=0)
    FRes = np.mean(np.array([0, results[indices[i]].x[82:]
                             for i in range(topPer)]), axis=0)
    print(DRes.shape, FRes.shape)
    sys.exit()

    # compute D and F and concentration profiles
    D, F = fp.computeDF(DRes, FRes, shape=segments)
    # computing WMatrix
    W = fp.WMatrixVar(D, F, N, deltaXX)
    # computing concentration profiles
    ccRes = fp.calcC(cc[:, 0], tt[j], W=W)

    # --------------------------- saving data ------------------------------- #
    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'concentrationExpRes.csv',
               np.concatenate((cc, ccRes), axis=1), delimiter=',')
    # saving averaged DF
    np.savetxt(savePath+'DF.csv', np.array([D, F]).T, delimiter=',')
    # saving Error of top 1% of runs
    np.savetxt(savePath+'minError.csv', Error[indices[:topPer]])


if __name__ == "__main__":
    main()
