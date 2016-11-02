# -*- coding: utf-8 -*-
import inputOutput as io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import matplotlib.cm as cmx
import scipy.interpolate as ip

live = False


def main():
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
            'Mucus/Ch1_Positive.csv')
    data = io.readData(path)
    xx = data[:, 0]
    cc = data[:, 1:]

    N = cc[0, :].size  # number of profiles
    CMax = np.max(cc)
    XMax = np.max(xx)
    s = [ip.UnivariateSpline(xx, cc[:, i], s=15) for i in range(N)]
    xs = np.linspace(xx[0], xx[-1], 100)
    cc = np.array([s[i](xs) for i in range(N)]).T
    xx = xs

    plt.axis([0, 590, 0, 12.3])
    cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=N)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    # plt.ion()
    Z = [[0, 0], [0, 0]]
    levels = np.arange(0, (N*10)/60, 10/60)
    CS3 = plt.contourf(Z, levels, cmap=cm)
    plt.axis([0, XMax, 0, CMax])

    if live:
        plt.ion()

    for i in range(N):
        colorVal = scalarMap.to_rgba(i)
        colorText = (
            'color: (%4.2f,%4.2f,%4.2f)' % (colorVal[0],
                                            colorVal[1], colorVal[2]))

        plt.plot(xx, cc[:, i], color=colorVal, label=colorText)
        if live:
            plt.pause(0.05)

    if live:
        while True:
            plt.pause(0.05)

    Cmap = plt.colorbar(CS3, orientation='horizontal')
    Cmap.set_label('Time [min]')
    plt.ylabel('Concentration [µM]')
    plt.xlabel('Distance [µm]')
    plt.savefig('/Users/AmanuelWK/Desktop/Profiles_Positive_smoothed.pdf')
    plt.show()

if __name__ == "__main__":
    main()
