import numpy as np
import DiffusionEq_Skin as sk
import FPModel as fp
import matplotlib.pyplot as plt
import sys
import scipy.constants as ct


# --------------------------- reading results ------------------------- #
if sys.platform == "darwin":  # folder for mac
    path = ('/Users/AmanuelWK/Dropbox/PhD/Projects/FokkerPlanckModeling/'
            'Skin/Data/')
elif sys.platform.startswith("linux"):  # folder for linux
    path = ('/home/amanuelwk/Dropbox/PhD/Projects/FokkerPlanckModeling/'
            'Skin/Data/')

D_py = np.loadtxt(path+'RobertsResults/rerun/python_DF.txt',
                  delimiter=',')[6:88, 0]
F_py = np.loadtxt(path+'RobertsResults/rerun/python_DF.txt',
                  delimiter=',')[6:88, 1]
D_mat = np.loadtxt(path+'RobertsResults/rerun/matlab_DF.txt',
                   delimiter=',')[6:88, 0]
F_mat = np.loadtxt(path+'RobertsResults/rerun/matlab_DF.txt',
                   delimiter=',')[6:88, 1]


cc_exp = np.array([np.concatenate((np.ones(7)*0.0025, np.zeros(95))),
                   np.loadtxt(path+'ExperimentalData/p10min.txt'),
                   np.loadtxt(path+'ExperimentalData/p100min.txt')[:80],
                   np.loadtxt(path+'ExperimentalData/p1000min.txt')[:80]])

# D, F profiles
df_py = np.concatenate((D_py, F_py))
df_mat = np.concatenate((D_mat, F_mat))
c0 = np.concatenate((np.ones(7)*0.0025, np.zeros(95)))
bestErr = 0.5705

tt = np.array([0, 600, 6000, 60000])  # t in seconds
# using roberts discretization
X2 = 1  # in epidermis 1µm
X1 = (400-(2.5*X2))/4.5  # 400µm in epidermins
X3 = (20000-(4.5*X2))/10.5  # 2cm in deeper skin layers
deltaX = np.array([X1, X2, X3])
# --------------------------- reading results ------------------------- #


# print(df_py[-2], df_py[-1])
# print(df_mat[-2], df_mat[-1])
#
# sys.exit()

kJMol = ct.Boltzmann*305*ct.Avogadro/1000
FMax = 25/kJMol  # max value for delta F in k_BT
deltaF = np.linspace(0, FMax, 100)

f_dermisPY = np.linspace(df_py[-2], df_py[-2]+FMax, 100)
dF_py = abs(df_py[-2]-df_py[-1])*kJMol
dF_mat = abs(df_mat[-2]-df_mat[-1])*kJMol
f_dermisMAT = np.linspace(df_mat[-2], df_mat[-2]+FMax, 100)
time = np.concatenate((np.zeros(1), np.logspace(0, 8, 99),
                       np.ones(1)*tt[-1])).astype(int)

err_py = []
err_mat = []
ccEpi_py = []
ccEpi_mat = []
for i in range(100):
    df_py[-1] = f_dermisPY[i]
    df_mat[-1] = f_dermisMAT[i]
    res_py, T_py = sk.resFun(df=df_py, cc=cc_exp, tt=tt,
                             deltaX=deltaX, debug=True,
                             verb=True)
    res_mat, T_mat = sk.resFun(df=df_mat, cc=cc_exp, tt=tt,
                               deltaX=deltaX, debug=True,
                               verb=True)
    cc_epi_py = [fp.calcC(c0, time[j], T=T_py)[7:87] for j in range(time.size)]
    ccEpi_py.append(np.array(cc_epi_py))
    cc_epi_mat = [fp.calcC(c0, time[j], T=T_py)[7:87] for j in range(time.size)]
    ccEpi_mat.append(np.array(cc_epi_mat))
    err_py.append(res_py)
    err_mat.append(res_mat)

# plotting error over delta F
plt.figure()
plt.title('Python: error')
plt.plot(deltaF, err_py, 'k-')
plt.plot(deltaF, np.ones(deltaF.size)*bestErr, 'k--')
plt.axvline(x=dF_py/kJMol, ymax=bestErr/1.5, c='k', ls='--')
plt.ylim([0, 1.5])
plt.xlabel('$\Delta$F [k$_B$T]')
plt.ylabel('$\sigma$ [µg/(cm$^2$µm)]')
# this is an inset axes over the main axes
plt.axes([.55, .55, .4, .3])
plt.plot(deltaF[-75:], err_py[-75:], 'k-')
plt.plot(deltaF[-75:], np.ones(deltaF.size)[-75:]*bestErr, 'k--')
plt.axvline(x=dF_py/kJMol, c='k', ls='--')
plt.xlabel('$\Delta$F [k$_B$T]')
plt.ylabel('$\sigma$ [µg/(cm$^2$µm)]')
# plt.show()
plt.savefig('/Users/AmanuelWK/Desktop/python_err.pdf')

# plotting 1000 min profiles for different delta F
col = ['b', 'c', 'g', 'r']
plt.figure()
timeIndex = np.abs(time-tt[-1]).argmin()
plt.title('Python: %i min profiles' % int(time[timeIndex]/60))
for j, i in enumerate([0, 25, 50, 75]):
    plt.plot(ccEpi_py[i][timeIndex, :], c=col[j], ls='-',
             label='$\Delta$F = %.2f k$_B$T' % deltaF[i])
plt.plot(cc_exp[-1], 'k-.', label='Experiment')
plt.xlabel('Skin depth [µm]')
plt.ylabel('c [µg/(cm$^2$µm)]')
plt.legend()
# plt.show()
plt.savefig('/Users/AmanuelWK/Desktop/python_con.pdf')

# plotting penetrated concentration in epidermis
plt.figure()
plt.title('Python: amount in epidermis')
for j, i in enumerate([0, 25, 50, 75]):
    plt.plot(time[:-1]+1, [np.sum(ccEpi_py[i][j, :])*80 for j in range(100)],
             c=col[j], ls='-', label='$\Delta$F = %.2f k$_B$T' % deltaF[i])
index = np.abs(f_dermisPY*kJMol-dF_py).argmin()
plt.plot(time[:-1]+1, [np.sum(ccEpi_py[index][j, :])*80 for j in range(100)],
         'k--', label='$\Delta$F = %.2f k$_B$T - result' % f_dermisPY[index])
plt.plot([1, 601, 6001, 60001],
         [np.sum(cc_exp[0][7:87])*80, np.sum(cc_exp[1])*73,
          np.sum(cc_exp[2])*80, np.sum(cc_exp[3])*80], 'ko')
plt.xlim(0.75, 1E8)
plt.legend()
plt.xlabel('Penetration time t+1 [s]')
plt.xscale('log')
plt.ylabel('c$_{epi}$ [µg/cm$^2$]')
# plt.show()
plt.savefig('/Users/AmanuelWK/Desktop/python_epi.pdf')


plt.figure()
plt.title('Matlab: error')
plt.plot(deltaF, err_mat, 'k-')
plt.plot(deltaF, np.ones(deltaF.size)*bestErr, 'k--')
plt.axvline(x=dF_mat/kJMol, ymax=bestErr/1.5, c='k', ls='--')
plt.ylim([0, 1.5])
plt.xlabel('$\Delta$F [k$_B$T]')
plt.ylabel('$\sigma$ [µg/(cm$^2$µm)]')
# this is an inset axes over the main axes
plt.axes([.55, .55, .4, .3])
plt.plot(deltaF, err_mat, 'k-')
plt.plot(deltaF, np.ones(deltaF.size)*bestErr, 'k--')
# plt.xlim(0, 10)
plt.axvline(x=dF_mat/kJMol, c='k', ls='--')
plt.xlabel('$\Delta$F [k$_B$T]')
plt.ylabel('$\sigma$ [µg/(cm$^2$µm)]')
plt.savefig('/Users/AmanuelWK/Desktop/matlab_err.pdf')


# plotting 1000 min profiles for different delta F
plt.figure()
plt.title('Matlab: %i min profiles' % int(time[timeIndex]/60))
for j, i in enumerate([0, 25, 50, 75]):
    plt.plot(ccEpi_mat[i][timeIndex, :], c=col[j], ls='-',
             label='$\Delta$F = %.2f k$_B$T' % deltaF[i])
plt.plot(cc_exp[-1], 'k-.', label='Experiment')
plt.xlabel('Skin depth [µm]')
plt.ylabel('c [µg/(cm$^2$µm)]')
plt.legend()
# plt.show()
plt.savefig('/Users/AmanuelWK/Desktop/matlab_con.pdf')


plt.figure()
plt.title('Matlab: amount in epidermis')
for j, i in enumerate([0, 25, 50, 75]):
    plt.plot(time[:-1]+1, [np.sum(ccEpi_mat[i][j, :])*80 for j in range(100)],
             c=col[j], ls='-', label='$\Delta$F = %.2f k$_B$T' % deltaF[i])
index = np.abs(f_dermisMAT*kJMol-dF_mat).argmin()
plt.plot(time[:-1]+1, [np.sum(ccEpi_mat[index][j, :])*80 for j in range(100)],
         'k--', label='$\Delta$F = %.2f k$_B$T - result' % f_dermisMAT[index])
plt.plot([1, 601, 6001, 60001],
         [np.sum(cc_exp[0][7:87])*80, np.sum(cc_exp[1])*73,
          np.sum(cc_exp[2])*80, np.sum(cc_exp[3])*80], 'ko')
plt.xlim(0.75, 1E8)
plt.legend()
plt.xlabel('Penetration time t+1 [s]')
plt.xscale('log')
plt.ylabel('c$_{epi}$ [µg/cm$^2$]')
plt.savefig('/Users/AmanuelWK/Desktop/matlab_epi.pdf')
