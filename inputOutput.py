# -*- coding: utf-8 -*-
import numpy as np
import csv
from matplotlib import pyplot as plt
import scipy.interpolate as ip
import scipy.signal as sg


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
        window = int(profiles[:, 0].size/4)
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
