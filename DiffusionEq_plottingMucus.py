import numpy as np
import inputOutput as io
import argparse as ap
import plottingScripts as ps
import os
import sys

'''
This script plots the analyzed data from the runs
'''


def main():
    # --------------- parsing command line inputs --------------------------- #
    parser = ap.ArgumentParser()
    parser.add_argument('path', help='define the relative path to '
                        'folder containing data files for plotting')
    parser.add_argument('name', help='define the name of the experimental'
                        'concentration profiles')
    parser.add_argument('-s', '--save', action='store_true', help='saves '
                        'plots and data to files to desktop')
    args = parser.parse_args()
    path = args.path
    name = args.name
    # folder to save figures in changes depending on system
    if sys.platform == "darwin":  # folder for mac
        figPath = '/Users/AmanuelWK/Desktop/%s/Figures/' % name
    elif sys.platform.startswith("linux"):  # folder for linux
        figPath = '/home/amanuelwk/Desktop/%s/Figures/' % name

    if args.save:
        if not os.path.exists(figPath):
            os.makedirs(figPath)

    # ------------------- experimental parameters ----------------------- #
    dt = 300  # time difference between c-profiles in s
    boundary = 100  # position of boundary between mucus and buffer in µm

    # ------------------- loading analyzed data ------------------------- #
    eData = io.readData(path+'Error.csv')
    DF = io.readData(path+'DF.csv')
    cData = io.readData(path+'concentrationExpRes.csv')
    # restData = np.load(path+'data.npy')
    # saving to seperate arrays
    distanceMuM, EMin, ESTD = eData.T
    D, F = DF.T
    # for more profiles analysis
    # xx, cc, ccRes = cData[:, 0], cData[:, 1:9], cData[:, 9:]
    # standart parameters
    xx, cc, ccRes = cData[:, 0], cData[:, 1:5], cData[:, 5:]

    M = cc[0, :].size  # number of different transition distances
    minD = distanceMuM[np.argmin(EMin)]  # optimal transition layer thickness
    TransIndex = np.argwhere(abs(xx - boundary) ==  # bin clostest to boundary
                             np.min(abs(xx - boundary)))[0, 0].astype(int)
    tt = np.arange(M)*dt  # vectors contains time for profile i

    # ---------------------- plotting figures ---------------------------- #
    # starting with Error over transition layer thickness
    ps.plotMinError(distanceMuM, EMin, ESTD, save=args.save,
                     path=figPath)
    # deprecated
    # ps.plotMinError(distanceMuM[1:], EMin[1:], ESTD[1:], save=args.save,
                    # path=figPath)
    # for plotting best D and F in the same figure
    ps.plotDF(xx, D, F, save=args.save, path=figPath)
    # plotting concentration profiles for best run
    ps.plotCon(xx, cc, ccRes, tt, TransIndex, layerD=minD, save=args.save,
               path=figPath)

    '''
    add option to plot other data from restData array
    '''


if __name__ == "__main__":
    main()
