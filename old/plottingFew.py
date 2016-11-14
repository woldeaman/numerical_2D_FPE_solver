# -*- coding: utf-8 -*-
import inputOutput as io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import matplotlib.cm as cmx
import scipy.interpolate as ip
import sys

live = False


def main():
    # path for work
    # path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
            # 'Mucus/Ch1_Positive.csv')
    # path for home
    path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
            'Mucus/Results/ExperimentalData/Ch1_Positive.csv')
    data = io.readData(path)
    xx = data[:, 0]
    cc = data[:, 1:]
    N = cc[0, :].size  # number of profiles
    tt = (np.arange(0, N)*10)/60

    CMax = np.max(cc)
    XMax = np.max(xx)
    s = [ip.UnivariateSpline(xx, cc[:, i], s=5) for i in range(N)]
    xs = np.linspace(xx[0], xx[-1], 100)
    ccFit = np.array([s[i](xs) for i in range(N)]).T
    xxFit = xs

    # plt.ion()
    cm = plt.get_cmap('rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=N)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    plt.axis([0, XMax, 0, CMax])

    for i in range(0, N, 30):
        colorVal = scalarMap.to_rgba(i)
        plt.plot(xx, cc[:, i], label=str(int(tt[i]))+' min',  color=colorVal)

    plt.legend()
    plt.ylabel('Concentration [µM]', fontsize=16)
    plt.xlabel('Distance [µm]', fontsize=16)
    plt.savefig('/home/amanuelwk/Desktop/profiles.pdf')
    plt.show()

if __name__ == "__main__":
    main()
