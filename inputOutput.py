# -*- coding: utf-8 -*-
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate as ip
import scipy.signal as sg
import argparse as ap
import sys


def startUp():
    '''
    This function reads input values from terminal and sets up everything
    to run the ls-optimization for the full DF profile analysis as well as
    for the two box model.
    '''

    # ---------------- parsing command line inputs ------------------------- #
    # gathering path to data and setting verbosity
    parser = ap.ArgumentParser(description=(
        """
        This script determines free energy and diffusivity profiles, based
        on supplied experimental concentration profiles. First column is always
        assumed to be z-distance vector! So far, only analysis with dx = const.
        Following 'i' columns are profiles at times dt*i.
        """), formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', dest='path', type=str,
                        help='Define the path to data for analysis. First '
                        'column in file should be distance vector, following '
                        'columns profiles at different time points')
    parser.add_argument('-v', dest='verbosity', type=int, help='set '
                        'verbosity level ranging from 0 - no output to 2 - '
                        'full output, -1 - means custom verbose mode')
    parser.add_argument('-pre', dest='pre', action='store_true',
                        help='Only look at raw data and pre-processing'
                        'results. Does not start analysis.')
    parser.add_argument('-ana', dest='analysis', action='store_true',
                        help='Do only plotting and analysis of previous run. '
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
    # select apropriate boundary conditions
    print('Analysis with fixed concentration at left boundary? [yes/no]')
    answer = input()
    yes, no = ['yes', 'y', 'Yes', 'YES'], ['no', 'n', 'No', 'NO']
    if answer in yes:
        bc_mode = 'open1side'  # analysis with fixed concentration
    elif answer in no:
        bc_mode = 'reflective'  # analysis with reflecting boundary conditions
    else:
        print('Error: input could not be read. Supply yes or no answer!')
        sys.exit()

    # reading fixed concentration if chosen bc
    if bc_mode is 'open1side':
        # 4µM for Kathy's data and and 15 µM for Tahouras data
        print('Set concentration value at boundary (same units as profiles):')
        c0 = float(sys.stdin.readline())
    else:
        c0 = None

    print('Set temporal resolution, supply dt in seconds:')
    dt = int(sys.stdin.readline())

    print('Choose profiles for analysis (supply timepoints in seconds,'
          '"all" means all profiles will be analyzed):')
    answer = input()
    if "all" in answer:
        tt = "all"
    else:
        tt = np.array([int(nbr) for nbr in answer.split()])

    print('Should profiles be pre-processed? If yes, profiles will be '
          'filtered using Savitzky-Golay. [yes/no]:')
    answer = input()
    if answer in yes:
        do_pre = True  # filter profiles
    elif answer in no:
        do_pre = False  # do not filter profiles
    else:
        print('Error: input could not be read. Supply yes or no answer!')
        sys.exit()

    print('Set regularization parameter alpha (zero means no regularization):')
    alpha = float(sys.stdin.readline())  # factor for L2 regularization

    print('Set number of analysis runs:')
    Runs = int(sys.stdin.readline())  # how many start D, Fs should be tried
    # ---------------- setting up analysis parameters ----------------------- #

    # -------------- reading and pre-processing profiles ------------------- #
    # reading profiles
    print('\nReading profiles...')
    try:  # change seperator accordingly
        data = readData(args.path, sep=';')
    except ValueError:
        try:
            data = readData(args.path, sep=',')
        except ValueError:
            data = readData(args.path, sep=' ')
    xx_exp = data[:, 0]  # first column assumed to be distance vector

    # now reading profiles based on input for different timepoints
    if "all" in tt:
        cc_exp = np.array(data[:, 1:])
        tt = np.arange(0, cc_exp[0, :].size*dt, dt)
    else:
        cc_exp = np.array([data[:, int(t/dt + 1)] for t in tt]).T

    if do_pre:
        # pre processing of profiles
        # filtering and setting negative c-values to zero
        print('\nDoing pre-processing...')
        xx, cc = preProcessing(xx_exp, cc_exp, window=5, order=3)
        np.savetxt('preProcessedProfiles.txt', np.c_[xx, cc], delimiter=',',
                   header='Profiles were smoothed using Savitzky-Golay filter'
                   ' \nCloumn 1: x-distance [micro meters]'
                   '\nColumn 2-5: c-profiles at t0-t3')
        print('Finished pre-processing and saved smoothed profiles.')
    else:
        # otherwise take profiles as they are
        xx, cc = xx_exp, cc_exp

    if args.pre:
        # plotting smoothed and original data as comparison
        print('\nDoing pre-processing only.')
        print('\nRaw data x-Axis ranges from Xmin = %2.f to Xmax = %2.f, '
              'with discretization deltaX = %2.f' % (np.min(xx_exp),
                                                     np.max(xx_exp),
                                                     (xx_exp[1]-xx_exp[0])))
        print('Assuming temporal discretization of deltaT = %is we have'
              ' data for %2.f minutes' % ((dt, (data[0, :].size-2)*dt/60)))

        # plotting profiles
        colors = [cm.jet(x) for x in np.linspace(0, 1, tt.size)]  # creating colormap
        for c_exp, c_smooth, c in zip(cc_exp.T, cc.T, colors):
            plt.plot(xx_exp, c_exp, '--', c=c)
            plt.plot(xx, c_smooth, '-', c=c)
        plt.show()
        sys.exit()

    deltaX = abs(xx[0] - xx[1])  # discretization width, assuming constant dx
    dim = cc[:, 0].size  # number of discretization bins
    # -------------- reading and pre-processing profiles ------------------- #

    # setting reasonable bounds for F e [-FBound, FBound], D e [0, DBound]
    # and setting number of runs
    DBound = 1000
    FBound = 20
    bndsDUpper = np.ones(dim)*DBound
    bndsFUpper = np.ones(dim)*FBound
    bndsDLower = np.zeros(dim)
    bndsFLower = np.ones(dim)*(-FBound)
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))
    FInit = 0
    DInit = (np.random.rand(dim, Runs)*DBound)

    print('\nStarting optimization...\n')
    return (bc_mode, dim, verbosity, Runs, ana, deltaX, c0, xx, cc, tt, bnds,
            FInit, DInit, alpha)


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
        # standart windows size is half of profile size
        window = int(profiles[:, 0].size/2)
        if window % 2 == 0:  # only odd values for window size work
            window = window + 1

    filtered = np.array([sg.savgol_filter(
        profiles[:, i], window, order, mode='nearest')
                         for i in range(profiles[0, :].size)]).T
    interpolated = [ip.UnivariateSpline(xx, filtered[:, i], s=0)
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


def tapeStripping(concentration, nbrTapeStrips, xx=None, species='human',
                  smoothing=0.025, plot=False):
    '''
    This function computes a concentration profile from measurements using
    tape strips, using heuristic formulas by Lademann.
    concentration -     array that contains the measured concentration
                        after each tapestrip
    nbrTapeStrips -     array that contains the number of tapestrips at which the
                        values in 'concentration' where measured
    xx            -     x-positions at which smoothing spline should be evaluated,
                        standart: depth of tapestripping
    species       -     defines the type of skin, 'human' or 'porcine', using different
                        formulas for skin depth, standart: human
    smoothing     -     sets the smoothing parameter for the fitting spline, standart: 0.025
    plot          -     plot fitted profile
    '''

    # settting correct parameters for different species, taken from Lademan papers
    if species is 'human':
        A, l = 107, 21  # parameters for function relating tapestrips to depth
        SC = 12.5  # in µm, average thickness of SC of different body parts
    elif species is 'porcine':
        A, l = 104, 27  # function parameters
        SC = 21  # thickness of SC in µm, for porcine ear
    else:
        print('\nUnknown species, choose either "human" or "porcine"\n\n')
        sys.exit()

    # differential concentration is what was in the tapestripped skin
    diff = np.array([concentration[i] - concentration[i+1]
                     for i in range(len(concentration)-1)])
    # skin depth for each tape strip
    depth = ((A - 111*np.exp(-nbrTapeStrips[1:]/l))/100)*SC
    # setting negative values to zero, NOTE: this is because of heuristic Lademann formula...
    depth = np.array([0 if x < 0 else x for x in depth])

    # if no x-vector is given, evaluate spline along entire tapestripping depth
    if xx is None:
        xx = np.linspace(depth[0], depth[-1], 15)
    spline = ip.UnivariateSpline(depth, diff, s=smoothing)

    if plot:
        x_plot = np.linspace(xx[0], xx[-1], 100)
        plt.figure()
        plt.plot(depth, diff, 'ko', label='ESR measurements')
        plt.plot(x_plot, spline(x_plot), 'k--', label='Cubic Smoothing Spline')
        plt.legend()
        plt.xlabel('z-distance [µm]')
        plt.ylabel('Concentration [nmol/cm$^2$]')
        # plt.savefig('/Users/AmanuelWK/Desktop/SC_profile_disrupted.pdf', bbox_inches='tight')
        plt.show()
        plt.close('all')

    return xx, spline(xx)
