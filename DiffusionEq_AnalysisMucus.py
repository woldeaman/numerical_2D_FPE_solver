import numpy as np
import inputOutput as io
import FPModel as fp
import scipy.special as sp
import argparse as ap
# import os
# for debugging
# import sys

'''
this script saves D, F and computed and experimental concentration profiles
together with transition layer thickness and error for each run
'''


def main():
    # parsing command line inputs
    parser = ap.ArgumentParser()
    parser.add_argument('path', help='define the relative path to '
                        'folder containing result.npy file')
    parser.add_argument('name', help='defines name of experimental'
                        ' concentration data for which analyis was performed')
    args = parser.parse_args()
    # gathering path to data and name of experiment
    path = args.path
    name = args.name+'.csv'

    # gathering experimental data from csv file in analysis folder
    Cdata = io.readData(path+name, sep=';')
    xx = Cdata[:, 0]  # first line in document is x-position
    # change number of profiles according to analysis type
    cc = np.array([Cdata[:, 1], Cdata[:, 31], Cdata[:, 61], Cdata[:, 91]]).T
    # pre processing of profiles as in optimization script
    xx, cc = io.preProcessing(xx, cc)
    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 4  # concentration of peptide solution in µM
    dim = cc[:, 0].size  # number of bins, chosen during pre processing
    M = cc[0, :].size  # number of profiles
    TransIndex = np.argwhere(abs(xx-100) ==
                             np.min(abs(xx - 100)))[0, 0].astype(int)
    # same conditions as for analysis need to be kept here
    segments = np.concatenate((np.ones(TransIndex+1)*0,
                               np.ones(dim-TransIndex)*1)).astype(int)
    '''change distances definition for newer simulations'''
    distances = np.arange(0, (2*TransIndex)+1, 2)
    '''change to distanceMuM = (distances-1)*deltaX, for newer simulations'''
    distanceMuM = np.concatenate((deltaX*np.ones(1), (distances[1:]-1)*deltaX))
    n = int(sp.binom(M, 2))  # binomial because of counting profile differences

    # loading result and extracting data to print out
    results = np.load(path+'result.npy')
    '''only needed for compatibality issues'''
    # results = results[:, :, 0]  # for compatibality with older version
    # for compatibality with buffer experiment
    # results = results.reshape((1, results.size))
    '''only needed for compatibality with older version'''
    K = results[:, 0].size  # number of different transition sizes
    I = results[0, :].size  # number of different initial conditions

    # gathering data from simulations
    # loading error values, factor two, because of cost function definition
    Error = np.array([[np.sqrt(results[k, i].cost*2/(dim*n)) for i in range(I)]
                      for k in range(K)])
    indices = np.argsort(Error)  # for sorting according to error
    # minimal error for each transition layer size
    EMin = np.array([np.min(Error[k, :]) for k in range(K)])
    ESTD = np.array([np.std(Error[k, :]) for k in range(K)])
    indexLayer = np.argmin(EMin)

    # gathering F and D and subsequently computing corresponding profiles
    DRes = np.array([[results[k, indices[k, i]].x[:2] for i in range(I)]
                     for k in range(K)])
    FRes = np.array([[np.array([0, results[k, indices[k, i]].x[-1]])
                      for i in range(I)]
                     for k in range(K)])

    # gather D and F and concentration profiles for best run
    DF = np.array(fp.computeDF(DRes[indexLayer, 0, :], FRes[indexLayer, 0, :],
                               shape=segments, mode='transition',
                               transiBin=TransIndex, dx=distances[indexLayer]))
    D = DF[0, :]
    F = DF[1, :]
    # computing WMatrix
    W = np.array(fp.WMatrix(D, F, bc='open1side',
                            deltaX=deltaX)[0])
    W10 = np.array(fp.WMatrix(D, F, bc='open1side',
                              deltaX=deltaX)[1])
    # computing concentration profiles
    ccRes = np.array([fp.calcC(cc[:, 0], tt[j], W=W, bc='open1side', W10=W10,
                               c0=c0) for j in range(M)]).T

    # saving error data for plotting
    desktopPath = '/Users/amanuelwk/Desktop/'
    np.savetxt(desktopPath+'Error.csv',
               np.array([distanceMuM, EMin, ESTD]).T, delimiter=',')
    # saving analyzed data for best results for plotting
    np.savetxt(desktopPath+'concentrationExpRes.csv',
               np.concatenate((xx.reshape((dim, 1)), cc, ccRes), axis=1),
               delimiter=',')
    np.savetxt(desktopPath+'DF.csv', np.array([D, F]).T, delimiter=',')

    # saving rest to npy array
    # format is: data[k, i, l] for layer distance k, top i run and with
    # l=0: error, l=1: D_sol, l=2: D_Muc, l=3:F_sol = 0, l=4: F_muc
    data = np.concatenate((Error.reshape((K, I, 1)), DRes, FRes), axis=2)
    np.save(desktopPath+'data.npy', data)

    '''for buffer experiment'''
    # segments = np.zeros(dim+1)  # needed for buffer experiment conditions
    # DF = np.array([[fp.computeDF(DRes[k, i, :], FRes[k, i, :],
    #                              shape=segments, mode='segments')
    #                 for i in range(N)] for k in range(K)])
    # distanceMuM = deltaX*np.ones(1)
    '''for buffer experiment'''

if __name__ == "__main__":
    main()
