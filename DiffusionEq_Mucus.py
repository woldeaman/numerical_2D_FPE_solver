# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import inputOutput as io
import FPModel as fp
# for debugging
import sys
# import matplotlib.pyplot as plt

startTime = time.time()
conservation = False
verbose = True


def main():
    # path for work
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            'Mucus/Results/ExperimentalData/Ch1_Positive.csv')
    # path for home
    # path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
    #         'Mucus/Results/ExperimentalData/Ch1_Positive.csv')

    # reading profiles and take only samples for 4 different time points
    data = io.readData(path)
    xx = data[:, 0]
    cc = np.array([data[:, 1], data[:, 31], data[:, 61], data[:, 91]]).T

    # pre processing of profiles
    xx, cc = io.preProcessing(xx, cc)
    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 4  # concentration of peptide solution in µM

    # plotting concentration profiles
    # io.plotCon(cc=cc, xx=xx, live=True)
    # sys.exit()

    # finding interface bin position and
    # storing it in order to give it to optimization
    TransIndex = np.argwhere(abs(xx-100) ==
                             np.min(abs(xx - 100)))[0, 0].astype(int)

    # setting reasonable bounds for F and D
    # changed for segmentation analysis
    bndsDUpper = np.ones(2)*1000
    bndsFUpper = np.ones(1)*20
    bndsDLower = np.zeros(2)
    bndsFLower = -np.ones(1)*20
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    # setting initial conditions
    # DInit = (np.random.rand(512)*400)+200

    FInit = -5
    # transition = np.arange(0, 37, 2)
    distances = np.arange(2*TransIndex, (2*TransIndex)+1, 1)
    print(distances)
    DInit = np.linspace(0, 400, 4)

    results = np.array([[fp.optimization(DRange=DInit[i]*np.ones(2),
                                         FRange=FInit*np.ones(1),
                                         Dist=distances[k], bnds=bnds, cc=cc,
                                         tt=tt, deltaX=deltaX,
                                         mode='mucusModel', c0=c0,
                                         debug=conservation, verb=verbose,
                                         transition=TransIndex)
                         for i in range(DInit.size)]
                        for k in range(distances.size)])

    np.save('result.npy', results)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
