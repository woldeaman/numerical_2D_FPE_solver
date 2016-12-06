# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import inputOutput as io
import FPModel as fp
# import scipy.interpolate as ip
# for debugging
import sys

startTime = time.time()
parallel = False
conservation = True
verbose = True


def main():
    # path for work
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            'Skin/Results/ExperimentalData/')
    # path for home
    # path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
    #         'Skin/Results/ExperimentalData/')

    # reading profiles
    cc = np.array([np.concatenate((np.ones(1), np.zeros(70))),
                   io.readData(path+'10min.txt'),
                   io.readData(path+'100min.txt'),
                   io.readData(path+'1000min.txt')]).T
    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    N = cc[:, 0].size  # number of bins
    # deltaX = 1E-6
    # plotting concentration
    # io.plotCon(cc[1:, :], live=True)
    # sys.exit()

    # setting bounds, D first and F second
    bndsDUpper = np.ones(N)*2000
    bndsFUpper = np.ones(N)*20
    bndsDLower = np.zeros(N)
    bndsFLower = -np.ones(N)*20
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    # setting initial conditions
    DInit = np.random.rand(3000)*1000
    # DInit = np.linspace(0, 350, 4)
    FInit = 5

    results = np.array([fp.optimization(DRange=np.concatenate((np.ones(15)*0.0001*DInit[i], np.ones(N-15)*DInit[i])),
                                        FRange=FInit*np.ones(N), bnds=bnds,
                                        cc=cc, tt=tt, bc='reflective',
                                        debug=conservation, verb=verbose)
                        for i in range(DInit.size)])

    np.save('result.npy', results)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
