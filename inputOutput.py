# -*- coding: utf-8 -*-
import numpy as np
import csv
from matplotlib import pyplot as plt
import scipy.interpolate as ip


# reading data
def readData(path, sep=',', typo=float):
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
                             for row in reader])
        else:
            data = np.array([row[0] for row in reader])

    return data.astype(typo)  # returns np array


def preProcessing(xx, cc, smoothing=True, order=5, bins=100):
    '''
    Function takes care of negative entries for concentration (set to zero)
    and additionally smoothes profiles or normalizes them (to be implemented).
    Profiles should be in format cc[:, i] for different times t_i
    '''
    profiles = cc

    # smoothing of concentration profiles
    if smoothing:
        s = [ip.UnivariateSpline(xx, profiles[:, i], s=order)
             for i in range(profiles[0, :].size)]
        xs = np.linspace(xx[0], xx[-1], bins)
        profiles = np.array([s[i](xs) for i in range(profiles[0, :].size)]).T

    # taking care of negative concentration values
    for i in range(profiles[:, 0].size):
        for j in range(profiles[0, :].size):
            if (profiles[i, j]) <= 0:
                profiles[i, j] = 0

    return xs, profiles


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
