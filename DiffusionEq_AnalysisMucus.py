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
together with transition layer thickness and error for each run
'''


def main():
    # --------------- parsing command line inputs --------------------------- #
    parser = ap.ArgumentParser()
    parser.add_argument('path', help='define the relative path to '
                        'folder containing result.npy file')
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
    Cdata = io.readData(path+name+'.csv', sep=',')  # change separator accordingly
    xx = Cdata[:, 0]  # first line in document is x-position
    # change number of profiles according to analysis type
    cc = np.array([Cdata[:, 1], Cdata[:, 31], Cdata[:, 61], Cdata[:, 91]]).T
    tt = np.array([0, 300, 600, 900])  # t in seconds
    # pre processing of profiles as in optimization script
    xx, cc = io.preProcessing(xx, cc)
    deltaX = abs(xx[0] - xx[1])
    c0 = 4  # concentration of peptide solution in µM
    dim = cc[:, 0].size  # number of bins, chosen during pre processing
    M = cc[0, :].size  # number of profiles
    TransIndex = np.argwhere(abs(xx-100) ==
                             np.min(abs(xx - 100)))[0, 0].astype(int)
    # same conditions as for analysis need to be kept here
    segments = np.concatenate((np.ones(TransIndex+1)*0,
                               np.ones(dim-TransIndex)*1)).astype(int)
    '''change distances definition for newer simulations'''
    distances = np.arange(2, (2*TransIndex)+1, step=2)
    # deprecated
    # distances = np.arange(0, (2*TransIndex)+1, 2)
    '''change to distanceMuM = (distances-1)*deltaX, for newer simulations'''
    distanceMuM = (distances-1)*deltaX
    # deprecated
    # distanceMuM = np.concatenate((deltaX*np.ones(1),
    #                               (distances[1:]-1)*deltaX))
    n = int(sp.binom(M, 2))  # binomial because of counting profile differences

    # -------------------------- loading results --------------------------- #
    results = np.load(path+'result.npy')
    # deprecated
    # results = results[:, :, 0]  # for compatibality with older version

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
    if args.dsol_const:  # added support for different computational model
        if "neg" in name:
            dsol = 101.35  # value of Dsol for negative peptide
        else:
            dsol = 99.46  # value of Dsol for positive peptide
        DRes = np.array([[np.array([dsol, results[k, indices[k, i]].x[0]])
                          for i in range(I)]
                         for k in range(K)])  # D-profile with dsol fixed
    else:
        DRes = np.array([[results[k, indices[k, i]].x[:2] for i in range(I)]
                         for k in range(K)])
    FRes = np.array([[np.array([0, results[k, indices[k, i]].x[-1]])
                      for i in range(I)]
                     for k in range(K)])

    # gather D and F and concentration profiles for best run
    # add special case for buffer only experiment (only one D fitted)
    if "buffer" not in name:
        DF = np.array(fp.computeDF(DRes[indexLayer, 0, :],
                                   FRes[indexLayer, 0, :], shape=segments,
                                   mode='transition', transiBin=TransIndex,
                                   dx=distances[indexLayer]))
    else:
        # needed for buffer experiment conditions
        segments = np.zeros(dim+1).astype(int)  # only one D
        FRes = np.zeros(FRes.shape)  # no F was fitted
        DF = np.array(fp.computeDF(DRes[0, 0, :], FRes[0, 0, :],
                                   shape=segments, mode='segments'))
        # distanceMuM = deltaX*np.ones(1)

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
    # --------------------------- saving data ------------------------------- #
    # saving error data for plotting
    np.savetxt(savePath+'Error.csv',
               np.array([distanceMuM, EMin, ESTD]).T, delimiter=',')
    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'concentrationExpRes.csv',
               np.concatenate((xx.reshape((dim, 1)), cc, ccRes), axis=1),
               delimiter=',')
    np.savetxt(savePath+'DF.csv', np.array([D, F]).T, delimiter=',')

    # saving rest to npy array
    # format is: data[k, i, l] for layer distance k, top i run and with
    # l=0: error, l=1: D_sol, l=2: D_Muc, l=3:F_sol = 0, l=4: F_muc
    data = np.concatenate((Error.reshape((K, I, 1)), DRes, FRes), axis=2)
    np.save(savePath+'data.npy', data)

    # saving best run data to excel spreadsheet
    D1 = DRes[indexLayer, :int(I/10), 0]  # top 10% of runs for D_sol
    D2 = DRes[indexLayer, :int(I/10), -1]  # top 10% of runs for D_muc
    F2 = FRes[indexLayer, :int(I/10), -1]  # top 10% of runs for F_muc
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
    worksheet.write('A2', '%.2f +/- %.2f' % (D1[0], np.std(D1)))
    worksheet.write('B2', '%.2f +/- %.2f' % (D2[0], np.std(D2)))
    worksheet.write('C2', '%.2f +/- %.2f' % (F2[0], np.std(F2)))
    worksheet.write('D2', '%.2f' % distanceMuM[indexLayer])
    worksheet.write('E2', '%.2f' % np.min(EMin))
    # adjusting cell widths
    worksheet.set_column(0, 5, len('minError [+/- µM]'))
    workbook.close()


if __name__ == "__main__":
    main()
