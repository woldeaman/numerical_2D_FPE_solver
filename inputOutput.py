# -*- coding: utf-8 -*-
import numpy as np
import csv
from matplotlib import pyplot as plt
import scipy.interpolate as ip
import scipy.signal as sg
import argparse as ap
import sys


def startUp(mode):
    '''
    This function reads input values from terminal and sets up everything
    to run the ls-optimization for the full DF profile analysis as well as
    for the two box model.
    '''

    # ---------------- parsing command line inputs ------------------------- #
    # gathering path to data and setting verbosity
    parser = ap.ArgumentParser()
    parser.add_argument('-p', dest='path', type=str,
                        help='define the path to data for analysis')
    parser.add_argument('-v', dest='verbosity', type=int, help='set '
                        'verbosity level ranging from 0 - no output to 2 - '
                        'full output, -1 - means custom verbose mode')
    parser.add_argument('-pre', dest='pre', action='store_true',
                        help='Only look at raw data and pre-processing'
                        'results. Does not start analysis.')
    parser.add_argument('-ana', dest='analysis', action='store_true',
                        help='Do only plotting and analysis of previous run.'
                        'Does not start main script.')

    args = parser.parse_args()
    ana = args.analysis
    if args.verbosity is None:
        verbosity = 0
    else:
        verbosity = args.verbosity
    # ---------------- parsing command line inputs ------------------------- #

    # ---------------- setting up analysis parameters ----------------------- #
    # reading run parameters from stdin
    # 4µM for Kathy's data and and 15 µM for Tahouras data
    print('Set concentration of peptide solution (same units as profiles):')
    c0 = int(sys.stdin.readline())
    # if mode is 'twoBox':
    #     print('Set position of buffer-mucus interface '
    #           '(same units as profiles):')
    #     xInter = float(sys.stdin.readline())
    print('Choose profiles for analysis '
          '(supply timepoints in seconds, assuming dt = 10s):')
    tt = np.array([int(nbr) for nbr in sys.stdin.readline().split()])
    dt = 10  # so far supplied with profiles for every ten seconds

    print('Set number of analysis runs:')
    Runs = int(sys.stdin.readline())
    # ---------------- setting up analysis parameters ----------------------- #

    # -------------- reading and pre-processing profiles ------------------- #
    # reading profiles and take only samples for 4 different time points
    try:  # change seperator accordingly
        data = readData(args.path, sep=';')
    except ValueError:
        data = readData(args.path, sep=',')

    # pre processing of profiles
    # filtering and setting negative c-values to zero
    print('\nReading data and starting pre-processing.')
    xx_exp = data[:, 0]
    # tt = np.array([0, 300, 600, 900])  # t in seconds
    # cc_exp = np.array([data[:, 1], data[:, 31], data[:, 61], data[:, 91]]).T
    # now reading profiles based on input for different timepoints
    cc_exp = np.array([data[:, int(t/dt + 1)] for t in tt]).T
    xx, cc = preProcessing(xx_exp, cc_exp, order=5)
    np.savetxt('preProcessedProfiles.txt', np.c_[xx, cc], delimiter=',',
               header='Profiles were smoothed using Savitzky-Golay filter'
               ' \nCloumn 1: x-distance [micro_m]'
               '\nColumn 2-5: c-profiles at t0-t3 [micro_M]')
    print('Finished pre-processing and saved smoothed profiles.')

    if args.pre:
        # plotting smoothed and original data as comparison
        print('\nDoing pre-processing only.')
        print('\nRaw data x-Axis ranges from Xmin = %2.f to Xmax = %2.f, '
              'with discretization deltaX = %2.f' % (np.min(xx_exp),
                                                     np.max(xx_exp),
                                                     (xx_exp[1]-xx_exp[0])))
        print('Assuming temporal discretization of deltaT = 10s we have'
              ' data for %2.f minutes' % ((data[0, :].size-2)/6))

        # plotting profiles
        plt.plot(xx_exp, cc_exp, '--', label='original')
        plt.plot(xx, cc, '-', label='smoothed')
        plt.show()
        sys.exit()

    dim = cc[:, 0].size  # number of discretization bins
    deltaX = abs(xx[0] - xx[1])  # discretization width
    # -------------- reading and pre-processing profiles ------------------- #

    # setting reasonable bounds for F and D
    # and setting number of runs
    DBound = 1000
    FBound = 20
    # parameters is one more than number of bins, because of c0 at boundary
    # same number for D and F, because F roams freely now
    # if mode is 'twoBox':
    #     dim = 1  # for two box model only two parameters
    #     # interface bin position
    #     TransIndex = np.argwhere(abs(xx - xInter) ==
    #                              np.min(abs(xx - xInter)))[0, 0].astype(int)
    #     # for the case of d = 2 we have instant jump,
    #     # because bin1 = D1, bin2 = D2
    #     distances = np.arange(2, (2*TransIndex)+1, step=int(TransIndex/10))

    bndsDUpper = np.ones(dim+1)*DBound
    bndsFUpper = np.ones(dim+1)*FBound
    bndsDLower = np.zeros(dim+1)
    bndsFLower = np.ones(dim+1)*(-FBound)
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))
    FInit = 0
    DInit = (np.random.rand(dim+1, Runs)*DBound)

    print('\nStarting optimization.\nDiscretization width is %.2f µm.'
          % deltaX)

    # if mode is 'twoBox':
    #     return (verbosity, Runs, ana, deltaX, c0, xInter, xx, cc, tt, bnds,
    #             FInit, DInit, distances, TransIndex)
    # else:
    return verbosity, Runs, ana, deltaX, c0, xx, cc, tt, bnds, FInit, DInit


# reading data
def readData(path, sep=',', typo=float, comChar='#'):
    '''
    Function reads data from delimited file in the path location,
    in which is columns are separated by sep and
    quote lines starting with quote are ignored.
    - Addable: quotechar and skiplines support
    '''

    multiCol = False
    # checking for 2D or 1D array
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=sep)
        for row in reader:
            if len(row) > 1:
                multiCol = True

    # gathering data
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=sep)
        if multiCol:
            data = np.array([[row[i] for i in range(len(row))]
                             for row in reader if not
                             any([row[0].startswith(comChar[i])
                                  for i in range(len(comChar))])])
        else:
            data = np.array([row[0] for row in reader if not
                             any([row[0].startswith(comChar[i])
                                  for i in range(len(comChar))])])

    return data.astype(typo)  # returns np array


def preProcessing(xx, cc, order=3, window=None, bins=100):
    '''
    Function takes care of negative entries for concentration (set to zero)
    and additionally smoothes profiles using a Savitzky-Golay filter
    Profiles should be in format cc[:, i] for different times t_i
    '''
    profiles = cc

    # taking care of negative concentration values prior to smoothing
    for i in range(profiles[:, 0].size):
        for j in range(profiles[0, :].size):
            if (profiles[i, j]) < 0:
                profiles[i, j] = 0

    # filtering/smoothing of concentration profiles
    if window is None:
        # standart windows size is quarter of profile size
        window = int(profiles[:, 0].size/2)
        if window % 2 == 0:  # only odd values for winow size work
            window = window + 1

    filtered = np.array([sg.savgol_filter(
        profiles[:, i], window, order, mode='nearest')
                         for i in range(profiles[0, :].size)]).T
    interpolated = [ip.UnivariateSpline(xx, filtered[:, i], s=0.5)
                    for i in range(filtered[0, :].size)]
    xs = np.linspace(xx[0], xx[-1], bins)
    profiles = np.array([interpolated[i](xs)
                         for i in range(profiles[0, :].size)]).T

    # taking care of negative concentration values after smoothing
    for i in range(profiles[:, 0].size):
        for j in range(profiles[0, :].size):
            if (profiles[i, j]) < 0:
                profiles[i, j] = 0

    return xs, profiles


# averaging c-profiles for new mucus data
def averageCon(xx, cc, xEnd=-1, tEnd=-1, dt=10):
    '''
    Computes the average concentration profiles from multiple data sets,
    where xx[i], cc[i][:, t] are arrays containing data for
    measurement i at time t. Averaged data is as long as shortest raw data.
    XEnd - x position [in µm] until which profiles should be analysed,
    if XEnd = -1 complete profiles will be averaged.
    tEnd - time point [in s] until which profiles sould be analysed,
    if tEnd = -1 complete profiles will be averaged.
    dt - Invervall at which profiles are recorded [standart is dt = 10s]
    '''

    samples = len(cc)  # number of samples
    if xEnd == -1:  # setting xEnd to maximal value for XEnd = -1
        xEnd = max([np.max(xx[i]) for i in range(samples)])
    if tEnd == -1:  # setting tEnd to maximal value for tEnd = -1
        tEnd = max([(cc[i][0, :].size)*dt for i in range(samples)])

    # t-vector for each sample
    tt = [np.arange(cc[i][0, :].size)*dt for i in range(samples)]
    dx = xx[0][1]  # assuming same bin width in all measurements

    # take minimum index to be able to average over all profiles
    xInd = np.min([np.argmin(abs(xx[i] - xEnd)) for i in range(samples)])
    tInd = np.min([np.argmin(abs(tt[i] - tEnd)) for i in range(samples)])

    # averaging profiles
    ccAver = np.array([[np.average([cc[i][x, t] for i in range(samples)
                                    if (cc[i][:, 0].size > x and
                                        cc[i][0, :].size > t)])
                        for t in range(tInd)]
                       for x in range(xInd)])

    xxAver = np.arange(xInd)*dx  # x-vector for averaged profile

    return xxAver, ccAver


# plotting concentration profiles for different times
def plotCon(cc, xx=None, live=False):
    '''
    Plotting concentration profiles, live if wanted for i profiles cc[:,i]
    '''

    # if no x vector is given just plot cc
    if xx is None:
        xx = np.array(range(cc[:, 0].size))

    CMax = np.max(cc)
    XMax = np.max(xx)

    if live:
        plt.axis([0, XMax, 0, CMax])
        plt.ion()

        for i in range(cc[0, :].size):
            plt.plot(xx, cc[:, i])
            plt.pause(0.05)

        while True:
            plt.pause(0.05)
    else:
        for i in range(cc[0, :].size):
            plt.axis([0, XMax, 0, CMax])
            plt.ylabel('Concentration [µM]')
            plt.xlabel('Distance [µm]')
            plt.plot(xx, cc[:, i])
            plt.show()
