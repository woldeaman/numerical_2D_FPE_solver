# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import functools as ft
import inputOutput as io
import FPModel as fp
from multiprocessing import Pool
import scipy.interpolate as ip
import sys
import matplotlib.pyplot as plt
import scipy.linalg as al
import numpy.linalg as la

startTime = time.time()
parallel = True


D = np.append(np.ones(2)*400, np.ones(1)*250)
F = np.append(np.ones(2)*15, np.ones(1)*10)

W, W10 = fp.WMatrixPart(D, F)
# W = np.array([[fp.WMatrixOld(i, j, D, F, 100) for i in range(100)] for j in range(100)])
# print(W.shape)
# sys.exit()

cc0 = np.append(np.ones(1)*4, np.zeros(24))
# print(cc0)
# sys.exit()

c0 = 4
tt = np.array([0, 10, 20, 30, 40, 50])
b = np.append(np.ones(1)*c0*W10, np.zeros(24))
Qb = np.dot(al.inv(W), b)
T = al.expm(W)

# print(np.dot(al.expm(W*tt[1]), cc0))
# ccRes = np.array([np.dot(al.expm(W*tt[i]), cc0) for i in range(tt.size)])
ccRes = np.array([fp.calcC(cc=cc0, t=tt[i], T=T, Qb=Qb, bc='open1side') for i in range(tt.size)])
ccRes = ccRes.T

xx = np.arange(ccRes[:, 0].size)

s = [ip.UnivariateSpline(xx, ccRes[:, i], s=5) for i in range(ccRes[0, :].size)]
xs = np.linspace(xx[0], xx[-1], 25)
ccRes = np.array([s[i](xs) for i in range(ccRes[0, :].size)]).T
xx = xs
for i in range(tt.size):
    plt.plot(ccRes[:, i])
    plt.show()

np.savetxt('cc.txt', ccRes, delimiter=', ')

bndsD = np.ones(3)*np.inf
bndsF = np.ones(3)*20
bnds = (np.zeros(2*(3)), np.concatenate((bndsD, bndsF)))

DInit = (np.random.rand(4)*150)+250
FInit = 10

optimize = ft.partial(fp.optimization, DRange=DInit, FRange=FInit,
                      bnds=bnds, cc=ccRes, tt=tt, bc='segmented', c0=c0,
                      debug=False, verb=True)


###########################
# linear and parallel implementation
###########################
if parallel:
    proc = Pool(processes=4)
    for i in proc.imap_unordered(optimize, range(DInit.size)):
        print('#%s: Time elapsed is %s s' % (i, time.time() - startTime))
        proc.close()
else:
    for i in range(DInit.size):
        optimize(i)


# io.plotCon(ccRes)

# for i in range(tt.size):
    # plt.axis([0, 100, 0, 4])
    # plt.plot(ccRes[i, :])
    # plt.show()
