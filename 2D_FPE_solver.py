# -*- coding: utf-8 -*-
# first start for fokker-planck equation in 2D
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import scipy.linalg as al
import numpy.linalg as la
from mpl_toolkits.mplot3d.axes3d import Axes3D  # needed for 3D plot support
from matplotlib import cm
startTime = time.time()


def plotting(X, Y, F, D, tt, cc, savePath=''):
    # plotting results
    fig = plt.figure()  # D and F in one figure
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, F, cmap=cm.coolwarm, antialiased=True)
    ax.set_title('Free Energy')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F [k$_B$T]')
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(X, Y, D, cmap=cm.coolwarm, antialiased=True)
    ax.set_title('Diffusivity')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('D [bins$^2$/timestep]')
    plt.savefig(savePath+'DF.pdf')

    cMax = np.max(cc)
    cMin = np.min(cc)

    M = tt.size
    fig = plt.figure()  # concentration profiles in second figures
    row = np.ceil(M/2)  # number of rows for c-plot
    col = np.floor(M/2)  # number of columns for c-plot

    for i in range(M):
        ax = fig.add_subplot(row, col, i+1, projection='3d')
        ax.plot_surface(X, Y, cc[i], cmap=cm.coolwarm, vmin=cMin, vmax=cMax/10, label='Original', antialiased=True)
        ax.set_title('t = %i' % tt[i])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Concentration')
        ax.set_zlim(cMin, cMax)
    plt.savefig(savePath+'profiles.pdf')


def WMatrix(D, F, dimY=None, dx=1):
    '''
    Function computes rate matrix for 2D systems, here the reduced index i
    is used defined as i = dimY*x + y, with dimY - number of bins in y-disc.
    D and F have to be reduced to 1D-index too in order for this to work!
    '''
    # using auxillary functions gFunc and fFunc as proposed by
    # Grima et. al - "Accurate discretization of advection-diffusion equations"
    # (2004) PRE https://doi.org/10.1103/PhysRevE.70.036703
    def g(i):
        return np.sqrt(D[i])*np.exp(F[i]/2)  # F in kB*T

    def f(i):
        return np.sqrt(D[i])*np.exp(-F[i]/2)  # F in kB*T
    # wRate(i --> j) = (f(j)*g(i))/(dx**2)

    N = D.size  # size of WMatrix is (dimX*dimY)x(dimX*dimY)
    if dimY is None:
        dimY = np.sqrt(N)  # equal discretization assumed if not specified

    #  first compute transition rates to come to bin i
    W = np.array([[f(i-1)*g(j-1)/(dx**2)
                   # only possible to go to neighbouring bins
                   if(
                       # nearest neighbours for first column
                       ((i-1) % dimY == 0 and
                        (abs(j-i) == dimY or
                         j == i+1 or
                         j == i+dimY+1 or
                         j == i-dimY+1))
                       or
                       # nearest neighbours for last column
                       (i % dimY == 0 and
                        (abs(j-i) == dimY or
                         j == i-1 or
                         j == i+dimY-1 or
                         j == i-dimY-1))
                       or
                       # nearest neighbours central bins
                       (((i-1) % dimY != 0 and i % dimY != 0) and
                        (abs(j-i) == dimY or
                         abs(j-i) == 1 or
                         abs(j-i-dimY) == 1 or
                         abs(j-i+dimY) == 1))
                       )
                   else 0
                   for j in range(1, N+1)] for i in range(1, N+1)])
    # indices are shifted because of modulo operator, doesn't handle 0 properly

    # then add rates to leave on main diagonal from original matrix
    for i in range(N):
        W[i, i] = -np.sum([W[j, i] for j in range(N)])

    if np.any(np.sum(W, 0) > 1E-10):
        print('WMatrix not row stochastic!')

    return W


def computeC(cc, tt, W=None, T=None, D=None, F=None, dimY=None, dx=1):
    '''
    Calculates concentration profiles at time t from W or T matrix with
    reflective boundaries, based on concentration profile cc
    '''
    # calculate only variables that are not given
    if T is None:
        if W is None:
            if (D is None) or (F is None):
                print('Error: You gotta give me something... no T, W, D or '
                      'F given!')
                sys.exit()
            else:
                W = WMatrix(D, F, dimY=dimY, dx=dx)
                T = al.expm(W)  # exponential of W
        else:
            T = al.expm(W)  # exponential of W

    return np.dot(la.matrix_power(T, tt), cc)


def main():
    # |- - - - - - > y             layout is in a way that y-axis points to
    # |(0, 0) (0, 1) ...           the right of matrix and x-axis points down,
    # |(1, 0) ...                  origin (0, 0) is in upper left corner,
    # | ...                        like a coordinate system tilted
    # v x                          at a right angle

    # defining grid and initial distribution
    dimX = 50
    dimY = 50
    cInit = 100
    c0 = np.zeros((dimX, dimY))
    c0[dimX//2-1:dimX//2+1, dimY//2-1:dimY//2+1] = cInit
    # meshgrid function needs X, Y in switchted order for correct ouput
    X, Y = np.meshgrid(np.arange(dimY), np.arange(dimX))

    # setting D and F
    D = np.ones((dimX, dimY))/100  # flat diffusivity profile
    F = np.zeros((dimX, dimY))  # flat energy landscape, normal diffusion
    # for i in range(dimY):  # for inclined energy profile
        # F[:, i] = -np.arange(dimX)

    # compute profiles from given d and f
    tt = np.array([0, 10, 100, 1000])
    
    cInput = [computeC(c0.reshape(c0.size), tt[i], D=D.reshape(D.size), F=F.reshape(F.size),
                       dimY=dimY).reshape(dimX, dimY) for i in range(tt.size)]
    # print([np.sum(cInput[i]) for i in range(len(cInput))])  # check for conservation of mass

    plotting(X, Y, F, D, tt, cInput)

if __name__ == "__main__":
    main()
    print("Execution time was %.2f minutes"
          % ((time.time() - startTime)/60))
