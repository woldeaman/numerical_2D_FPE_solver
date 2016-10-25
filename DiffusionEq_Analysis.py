import matplotlib.pyplot as plt
import csv
import numpy as np
from DiffusionEq import WMatrix
import scipy.linalg as al


# reading data
def readData(path, sep=','):
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

    return data  # returns np array


def main():
    save = True
    # printing results
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/DiffusionModel/'
            'Skin/Results/ComputedData/Run4_random/')

    # for parallelized version
    # gathering data and sorting according to error
    # M = 300  # number of runs to go through
    #
    # data = np.array([readData(path+'RunInfo/info_%s.csv' % i)[1, :]
    #                  for i in range(M)], dtype=float)
    #
    # DValue, EValue = data[1:, 1], data[1:, 2]
    # indices = np.argsort(EValue)

    # for linear version
    data = readData(path+'info.csv')

    DValue, EValue = data[1:, 1].astype(float), data[1:, 2].astype(float)
    indices = np.argsort(EValue)

    # Finding Top N runs
    N = 5
    F = np.array([readData(path+'DAndF/F_%s.txt' % str(indices[i]+1))
                  for i in range(N)], dtype=float)
    D = np.array([readData(path+'DAndF/D_%s.txt' % str(indices[i]+1))
                  for i in range(N)], dtype=float)
    cc = readData(path+'cc_1.csv').astype(float)

    dim = len(cc[0, :])
    x = np.arange(len(cc[0, :]))
    tt = np.array([0, 600, 6000, 60000])

    W = np.array([[[WMatrix(i, j, D[k, :], F[k, :], dim) for i in range(dim)]
                   for j in range(dim)] for k in range(N)])

    ccRes = np.array([[np.dot(al.expm((W[i, :, :].T)*tt[0]), cc[0, :]),
                      np.dot(al.expm((W[i, :, :].T)*tt[1]), cc[0, :]),
                      np.dot(al.expm((W[i, :, :].T)*tt[2]), cc[0, :]),
                      np.dot(al.expm((W[i, :, :].T)*tt[3]), cc[0, :])]
                      for i in range(N)])

    print('Top %s Runs with minimal error are: \n' % N)
    for i in range(N):
        print('Run #%s with E = %s \n'
              % (str(indices[i]+1), EValue[indices[i]]))

    # plotting results
    for i in range(N):
        plt.figure(i)
        plt.plot(D[i, :])
        plt.title('D_%s' % str(indices[i]+1))
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_D.pdf'
                        % str(indices[i]+1))
        plt.show()

        plt.figure(i+1)
        plt.plot(F[i, :])
        plt.title('F_%s' % str(indices[i]+1))
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_F.pdf'
                        % str(indices[i]+1))
        plt.show()

        plt.figure(i+2)
        expData0, = plt.plot(x[2:], cc[0, 2:], 'b--', label='0m Experiment')
        calcData0, = plt.plot(x[2:], ccRes[i, 0, 2:],
                              'b-', label='0m Computed')
        expData1, = plt.plot(x[2:], cc[1, 2:], 'r--', label='10m Experiment')
        calcData1, = plt.plot(x[2:], ccRes[i, 1, 2:],
                              'r-', label='10m Computed')
        expData2, = plt.plot(x[2:], cc[2, 2:], 'g--', label='100m Experiment')
        calcData2, = plt.plot(x[2:], ccRes[i, 2, 2:],
                              'g-', label='100m Computed')
        expData3, = plt.plot(x[2:], cc[3, 2:], 'k--', label='1000m Experiment')
        calcData3, = plt.plot(x[2:], ccRes[i, 3, 2:],
                              'k-', label='1000m Computed')

        plt.title('C-Profiles from run #%s' % (str(indices[i]+1)))
        plt.legend(handles=[expData0, calcData0, expData1, calcData1,
                            expData2, calcData2, expData3, calcData3])
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_profiles.pdf'
                        % str(indices[i]+1))
        plt.show()

if __name__ == "__main__":
    main()
