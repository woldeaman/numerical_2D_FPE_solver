import numpy as np
import inputOutput as io
import DiffusionEq_Skin_RobertsDiscretization as sk
import FPModel as fp
import sys


def main():
    # --------------------------- reading profiles ------------------------- #
    if sys.platform == "darwin":  # folder for mac
        path = ('/Users/AmanuelWK/Dropbox/PhD/Projects/FokkerPlanckModeling/'
                'Skin/Data/')
    elif sys.platform.startswith("linux"):  # folder for linux
        path = ('/home/amanuelwk/Dropbox/PhD/Projects/FokkerPlanckModeling/'
                'Skin/Data/')

    # # Roberts discretization
    cc = np.array([np.concatenate((np.ones(7)*0.0025, np.zeros(95))),
                   io.readData(path+'ExperimentalData/p10min.txt')[:73],
                   io.readData(path+'ExperimentalData/p100min.txt')[:80],
                   io.readData(path+'ExperimentalData/p1000min.txt')[:80]]).T

    # my original discretization
    # cc = np.array([np.concatenate((np.ones(10)*0.0025, np.zeros(90))),
    #               io.readData(path+'ExperimentalData/p10min.txt')[:73],
    #               io.readData(path+'ExperimentalData/p100min.txt')[:80],
    #               io.readData(path+'ExperimentalData/p1000min.txt')[:80]]).T
    #
    # X2 = 1  # discretization width in epidermis is 1µm
    # X1 = (400-(3.5*X2))/6.5  # transition between discretizations at bin 7
    # X3 = (20000-(3.5*X2))/6.5  # transition between discretizations at bin 83

    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    # using roberts discretization
    X2 = 1  # in epidermis 1µm
    X1 = (400-(2.5*X2))/4.5  # 400µm in epidermins
    X3 = (20000-(4.5*X2))/10.5  # 2cm in deeper skin layers
    deltaX = np.array([X1, X2, X3])
    # --------------------------- reading profiles -------------------------- #

    # read roberts results
    D = np.loadtxt(path+'RobertsResults/D.txt')
    F = np.loadtxt(path+'RobertsResults/F.txt')
    # read my results
    # dat = np.loadtxt('/Users/AmanuelWK/Desktop/Cluster/jobs/fokkerPlanckModel/'
    #                  'skin/0_RobertsDiscretization/results/DF.txt',
    #                  delimiter=',')
    # D = dat[:, 0]
    # F = dat[:, 2]
    df = np.concatenate((D[6:88], F[6:88]))

    res = sk.resFun(df=df, cc=cc, tt=tt, deltaX=deltaX, debug=True, verb=True)


if __name__ == "__main__":
    main()
