# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D with reflective BCs
# and minimization technique to obtain D and F profiles from c(ti)
# distributions at different times ti

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as al
import scipy.optimize as op
import scipy.special as sp
import time
import sys
import csv

startTime = time.time()


# reading concentration profiles
def readProfiles(path):
    numLines = sum(1 for line in open(path, 'r'))
    profile = np.zeros(numLines)
    i = 0
    with open(path, 'r') as file:
        # define file format using delimiters etc.
        reader = csv.reader(file, delimiter='\n', quotechar='#')
        for row in reader:
            profile[i] = float(row[0])
            i = i+1
    return profile


# define rate matrix recursively
# i = row index, j = column index
def WMatrix(i, j, d, f, dim):
    '''
    Calculates entries of rate matrix W with rank dim for each pair of
    indices i,j; dim = number of bins
    definition in R. Schulz, PNAS, 2016
    '''

    if (i == j):  # diagonal elements are defined recursively
        if (i == 0):
            value = -WMatrix(i+1, i, d, f, dim)
            # reflective BCs for start and end
        elif (i == dim-1):  # Number of bins N but index runs from 0,..,N-1
            value = -WMatrix(i-1, i, d, f, dim)
        else:
            value = -WMatrix(i-1, i, d, f, dim)-WMatrix(i+1, i, d, f, dim)

    elif (i == j-1 or i == j+1):  # matrix elements next to diagonal
        value = ((d[i]+d[j])/2)*np.exp(-(f[i]-f[j])/(2))

    else:  # everything further of than +/- 1 from diagonal is zero
        value = 0

    return value


# generally define error functional E
def EFunc(d, f, dim, cc, tt):
    '''
    cc and tt arrays with concentration profiles cc[i,:]
    for time tt[i] and tt[j] > tt[i] if j > i
    were cc is 2D array with cc.shape = (number of samples, number of bins)
    '''

    W = np.array([[WMatrix(i, j, d, f, dim)
                   for i in range(dim)] for j in range(dim)])

    M = cc[:, 1].size  # number of concentration profiles

    # check for detailed balance
    if np.any(np.sum(W, 1) != 0):
        rows = W[np.sum(W, 1) != 0, :]  # finding rows were row sum != 0
        check = max(np.sum(rows, 1)) > abs(W[1, 1])*1E-3
        # threshold: numerical error has to be
        # 1000 times smaller than first value of W
        if check:
            print('Error: Detailed balance is not obeyed, in rows: ',
                  np.nonzero(np.sum(W, 1) != 0))
            print(np.sum(W, 1))
            sys.exit()

    # filling up residual vector
    RR = np.zeros((int(sp.binom(M, 2)), dim))
    k = 0
    for j in range(M):
        for i in range(M):
            if j > i:
                RR[k, :] = np.array(cc[j, :] - np.dot(al.expm(
                    W*(tt[j] - tt[i])),  cc[i, :]))
                k += 1

    # calculating norm and functional to minimize
    RRn = np.array([al.norm(RR[i, :]) for i in range(RR[:, 1].size)])
    # E = (1/(dim*(M-1)))*np.sum(RRn**2)

    return (RRn*2)/(dim*(M-1))


def main():
    # reading profiles
    path = '/Users/AmanuelWK/GoogleDrive/PhD/Projects/DiffusionModel/Skin/'
    cc = np.array([readProfiles(path+'10min.txt'),
                   readProfiles(path+'100min.txt'),
                   readProfiles(path+'1000min.txt')])
    N = cc[1, :].size
    cc0 = np.append(1, np.zeros(N-1))  # initial concentration profile
    cc = np.insert(cc, 0, cc0, axis=0)
    # t in seconds
    tt = np.array([0, 600, 6000, 60000])

    def EMin(df):
        # function only depends on D and F
        # --> optimization with regard to D and F
        EM = EFunc(df[0:N], df[N:2*N], N, cc, tt)
        return EM

    # # setting inital values and bounds, D first and F second
    # initVal = np.append(np.ones(N)*50, np.ones(N)*50)
    # bnds = (np.zeros(2*N), np.ones(2*N)*np.inf)
    # print(result)
    # D = result.x[:N]
    # F = result.x[N:]
    # np.savetxt('D.txt', D, delimiter=', ')
    # np.savetxt('F.txt', F, delimiter=', ')

    # checking and saving concentration profiles of result
    F = np.append(np.ones(5)*4, np.ones(N-5))
    D = np.concatenate((np.ones(5)*100, np.ones(10)*10, np.ones(N-15)*100))

    print(F.shape)
    print(D.shape)

    W = np.array([[WMatrix(i, j, D, F, N)
                   for i in range(N)] for j in range(N)])
    ccRes = np.array([np.dot(al.expm(W*tt[0]), cc0),
                      np.dot(al.expm(W*tt[1]), cc0),
                      np.dot(al.expm(W*tt[2]), cc0),
                      np.dot(al.expm(W*tt[3]), cc0)])

    np.savetxt('ccRes.csv', ccRes, delimiter=', ')
    np.savetxt('cc.csv', cc, delimiter=', ')

    # printing results
    plt.figure(1)
    plt.plot(D)
    plt.title('Result for D')
    plt.show()

    plt.figure(2)
    plt.plot(F)
    plt.title('Result for F')
    plt.show()

    plt.figure(3)
    plt.plot(cc[0, :], 'ro--', ccRes[0, :], 'bx--')
    plt.title('C-Profile for t = %s s' % tt[0])
    plt.legend(('measured', 'calculated'))
    plt.show()

    plt.figure(4)
    plt.plot(cc[1, :], 'ro--', ccRes[1, :], 'bx--')
    plt.title('C-Profile for t = %s s' % tt[1])
    plt.legend(('measured', 'calculated'))
    plt.show()

    plt.figure(5)
    plt.plot(cc[2, :], 'ro--', ccRes[2, :], 'bx--')
    plt.title('C-Profile for t = %s s' % tt[2])
    plt.legend(('measured', 'calculated'))
    plt.show()

    plt.figure(6)
    plt.plot(cc[3, :], 'ro--', ccRes[3, :], 'bx--')
    plt.title('C-Profile for t = %s s' % tt[2])
    plt.legend(('measured', 'calculated'))
    plt.show()


if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
    
