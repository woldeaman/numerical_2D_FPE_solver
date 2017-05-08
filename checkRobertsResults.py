import numpy as np
import inputOutput as io
import DiffusionEq_Skin as sk


def main():
    # --------------------------- reading profiles ------------------------- #
    path = ('/Users/AmanuelWK/Dropbox/PhD/Projects/FokkerPlanckModeling/'
            'Skin/Data/')
    cc = np.array([np.concatenate((np.ones(10)*0.0025, np.zeros(90))),
                   io.readData(path+'ExperimentalData/p10min.txt')[:73],
                   io.readData(path+'ExperimentalData/p100min.txt')[:80],
                   io.readData(path+'ExperimentalData/p1000min.txt')[:80]]).T

    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    # computing discretization lengths
    X2 = 1  # discretization length in epidermis is 1µm
    X1 = (400-(3.5*X2))/6.5  # transition between discretizations at bin 7
    X3 = (20000-(3.5*X2))/6.5  # transition between discretizations at bin 83
    deltaX = np.array([X1, X2, X3])
    # --------------------------- reading profiles -------------------------- #

    # read roberts results
    D = np.loadtxt(path+'RobertsResults/D.txt')
    F = np.loadtxt(path+'RobertsResults/F.txt')
    df = np.concatenate((D[6:88], F[7:88]))

    sk.resFun(df=df, cc=cc, tt=tt, deltaX=deltaX, verb=True)


if __name__ == "__main__":
    main()
