import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from Model.GP_het_scedat import GP_hetscedat_class

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
y_var = np.load('../test_data/2D_data/output_reflect_var.npy').flatten()

############################################################################
# Set GP model
############################################################################

# Set kernels
kern=['MATERN_5_2_NS']
kern_ops = []
kern_var=['EXP_NS']
kern_var_ops = []

# Set output warp models
ow_model=['nat_log', 'meanstd']
ow_noise=['nat_log', 'meanstd']

# Input warp 
iw = True

# set class
gp =  GP_hetscedat_class(X, y, y_var, kern, kern_ops,  kern_var, kern_var_ops, iw, ow_model, ow_noise)

# set test and train split
gp.set_test_train(train_mean=0.7, train_noise=0.7)

############################################################################
# Noise Model
############################################################################

noise_gp_file = None

if noise_gp_file is not None:
    theta = np.load(noise_gp_file)
    gp.set_theta(theta, model='noise')
else:
    # Optimise noise gp
     gp.optimise_gp(model='noise', solver='opt', n_restarts=5)

print(gp.theta_noise)

# Test train plots
gp.test_train_plots(model='noise', fname='noise_plot.png')


############################################################################
# Main Model
############################################################################

mean_gp_file = None

if mean_gp_file is not None:
    theta = np.load(mean_gp_file)
    gp.set_theta(theta, model='mean')
else:
    # Optimise noise gp
     gp.optimise_gp(model='mean', solver='opt', n_restarts=5)

print(gp.theta)

# Test train plots
gp.test_train_plots(model='mean', fname='mean_plot.png')

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

Y_300, var_epi_300, var_noise_300 = gp.posterior_predict(X_300, model='mean', scale=True, get_var=True)
Y_500, var_epi_500, var_noise_500 = gp.posterior_predict(X_500, model='mean', scale=True, get_var=True)
Y_1000, var_epi_1000, var_noise_1000 = gp.posterior_predict(X_1000, model='mean', scale=True, get_var=True)

err_300 = 2.0*np.sqrt(var_epi_300 + var_noise_300)
err_500 = 2.0*np.sqrt(var_epi_500 + var_noise_500)
err_1000 = 2.0*np.sqrt(var_epi_300 + var_noise_1000)


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