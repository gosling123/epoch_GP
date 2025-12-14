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

X = np.load('../test_data/1D_data/input_mean.npy')
y = np.load('../test_data/1D_data/output_reflect_mean.npy').flatten()
y_var = np.load('../test_data/1D_data/output_reflect_var.npy').flatten()
X_obs = np.load('../test_data/1D_data/input_obs.npy')
y_obs = np.load('../test_data/1D_data/output_reflect_obs.npy')


############################################################################
# Set GP model
############################################################################

# Set kernels
kern=['RBF']
kern_ops = "k_1"
kern_var=['RBF']
kern_var_ops = "k_1"

# Set output warp models
ow_model=['nat_log', 'unit_var', 'sinharcsinh', 'affine' , 'sinharcsinh', 'meanstd']
ow_noise=['nat_log', 'unit_var', 'sinharcsinh', 'affine' , 'sinharcsinh', 'meanstd']

# Input warp 
iw = True

# set class
gp =  GP_hetscedat_class(X, y, y_var, kern, kern_ops,  kern_var, kern_var_ops, iw, ow_model, ow_noise)

# set test and train split
gp.set_test_train(train_mean=0.75, train_noise=0.75)


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
print(gp.theta_noise_labels)

# Test train plots
gp.test_train_plots(model='noise', fname='1D_noise_plot.png')


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
print(gp.theta_labels)

# Test train plots
gp.test_train_plots(model='mean', fname='1D_mean_plot.png')

############################################################################
# GP model prediction (NOISE)
############################################################################

# Extract noise model prediction
X_new = np.linspace(1e14, 1e16, 100)
noise_mean, noise_var = gp.posterior_predict(X_new, model='noise', scale=True, get_var=True)
noise_err = 2.0 * np.sqrt(noise_var)
lower = np.maximum(0, noise_mean - noise_err)
upper = noise_mean + noise_err


fig = plt.figure()
plt.scatter(X, y_var, c='b')
plt.scatter(gp.rescale_X(gp.X_var_train), gp.y_var_train, c='r')
plt.plot(X_new, noise_mean, c='blue')
plt.fill_between(X_new.flatten(), lower, upper, alpha=0.4, color='blue')
plt.xscale('log')
plt.savefig('var_ow.png')

############################################################################
# GP model prediction (MAIN MODEL)
############################################################################

# Extract model prediction
X_new = np.linspace(1e14, 1e16, 100)
mean, var_epi, var_noise = gp.posterior_predict(X_new, model='mean', scale=True, get_var=True)
err_epi = 2.0 * np.sqrt(var_epi)
err_tot = 2.0 * np.sqrt(var_epi + var_noise)
lower_epi = np.maximum(0, mean - err_epi)
upper_epi = mean + err_epi
lower_tot = np.maximum(0, mean - err_tot)
upper_tot = mean + err_tot

fig = plt.figure()

plt.plot(X_new, mean*100, c='orange', label=r'GP Median')
plt.fill_between(X_new, lower_epi*100, upper_epi*100, alpha=0.2, color='orange', label=r'Epistemic 95 $\%$ CI')
plt.fill_between(X_new, upper_epi*100, upper_tot*100, alpha=0.2, color='green', label=r'Total 95 $\%$ CI')
plt.fill_between(X_new, lower_tot*100, lower_epi*100, alpha=0.2, color='green')

plt.tick_params(axis="x", which='minor', direction="in", length=6, width=1.5)
plt.tick_params(axis="y", which='minor', direction="in", length=6, width=1.5)
plt.tick_params(axis="x", which='major', direction="in", length=12, width=3)
plt.tick_params(axis="y", which='major', direction="in", length=12, width=3)
plt.scatter(X_obs, y_obs*100, c='green', marker='x', alpha=0.7, s=30, label=r'EPOCH Observations')
plt.scatter(X, y*100, c='black', s=40, label = r'EPOCH Means')
plt.legend(frameon=False, loc='upper left')


plt.ylabel(r'$R_{SRS} \, \left(\%\right)$')
plt.xlabel(r'$I_0$ (Wcm$^{-2}$)')
plt.xscale('log')

plt.ylim(0, 40)
plt.savefig('1D_output_model.png')
