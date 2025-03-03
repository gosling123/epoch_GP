import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from Model.GP_model import GP_class

# Plotting Params
plt.rcParams["figure.figsize"] = (15,6)
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['image.cmap'] = 'jet'

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["mathtext.fontset"] = 'cm'

############################################################################
# Set Data
############################################################################

# 2D test Data
X = np.load('../test_data/2D_data/input_mean.npy')
y = np.load('../test_data/2D_data/output_reflect_mean.npy').flatten()


############################################################################
# Set GP model
############################################################################

# Set kernel
kern=['MATERN_5_2_ARD']
kern_ops = []

# Set output warp
ow_model=['nat_log', 'meanstd']

# Input warp 
iw = True

# set class
gp = GP_class(X, y, kern, kern_ops, iw, ow_model)


############################################################################
# GP model train
############################################################################

# GP file (from previous training)
gp_file = None

if gp_file is not None:
    gp.read_gp_model(gp_file)
else:
    # Set test train data
    gp.set_test_train(train_frac=0.75)
    # Optimise noise gp
    gp.optimise_gp(solver='opt', n_restarts=6, save=True, fname="2D_test.pkl")
print(gp.theta)
print(gp.theta_labels)

# Test train plots
gp.test_train_plots(fname='2D_test_train.png')

############################################################################
# GP model prediction
############################################################################

N = 5000
Ln_300 = np.ones(N)*300e-6
Ln_500 = np.ones(N)*500e-6
Ln_1000 = np.ones(N)*1000e-6
I = np.geomspace(1e14, 1.0e16, N)
X_300 = []
X_500 = []
X_1000 = []
for i in range(N):
    X_300.append([I[i], Ln_300[i]])
    X_500.append([I[i], Ln_500[i]])
    X_1000.append([I[i], Ln_1000[i]])

X_300 = np.array(X_300)
X_500 = np.array(X_500)
X_1000 = np.array(X_1000)

# Posterior predictions
Y_300, var_300 = gp.posterior_predict(X_300, scale=True, get_var=True)
Y_500, var_500 = gp.posterior_predict(X_500, scale=True, get_var=True)
Y_1000, var_1000 = gp.posterior_predict(X_1000, scale=True, get_var=True)

# Error estimation
err_300 = 2.0*np.sqrt(var_300)
err_500 = 2.0*np.sqrt(var_500)
err_1000 = 2.0*np.sqrt(var_300)


############################################################################
# Plot prediction
############################################################################

fig = plt.figure()

plt.loglog(X_300[:,0].flatten(), Y_300.flatten()*X_300[:,0].flatten(), color = 'blue', label=r'$L_n$ = 300 $\mu$m')
plt.plot(X_500[:,0].flatten(), Y_500.flatten()*X_500[:,0].flatten(), color = 'red', label=r'$L_n$ = 500 $\mu$m')
plt.plot(X_1000[:,0].flatten(), Y_1000.flatten()*X_1000[:,0].flatten(), color = 'green', label=r'$L_n$ = 1000 $\mu$m')

plt.xlim(1e14, 1e16)
plt.fill_between(X_300[:,0].flatten(), np.maximum((Y_300.flatten()-err_300)*X_300[:,0].flatten(),0.2*Y_300.flatten()*X_300[:,0].flatten()), (Y_300+err_300)*X_300[:,0].flatten(), alpha = 0.1, color = 'blue')
plt.fill_between(X_500[:,0].flatten(), np.maximum((Y_500.flatten()-err_500)*X_500[:,0].flatten(), 0.2*Y_500.flatten()*X_500[:,0].flatten()), (Y_500.flatten()+err_500)*X_500[:,0].flatten(), alpha = 0.1, color = 'red')
plt.fill_between(X_1000[:,0].flatten(), np.maximum((Y_1000.flatten()-err_1000)*X_1000[:,0].flatten(), 0.2*Y_1000.flatten()*X_1000[:,0].flatten()), (Y_1000.flatten()+err_1000)*X_1000[:,0].flatten(), alpha = 0.1, color = 'green')
plt.legend(frameon=False)


plt.tick_params(axis="x", which='minor', direction="in", length=6, width=1.5)
plt.tick_params(axis="y", which='minor', direction="in", length=6, width=1.5)
plt.tick_params(axis="x", which='major', direction="in", length=12, width=3)
plt.tick_params(axis="y", which='major', direction="in", length=12, width=3)

plt.ylabel(r'$I_{SRS}$ (Wcm$^{-2}$)')
plt.xlabel(r'$I_0$ (Wcm$^{-2}$)')
plt.savefig('2D_prediction.png')