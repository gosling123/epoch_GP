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
X = np.load('./test_data/2D_data/input_mean.npy')
y = np.load('./test_data/2D_data/output_reflect_mean.npy').flatten()
y_var = np.load('./test_data/2D_data/output_reflect_var.npy').flatten()

############################################################################
# Set GP model
############################################################################

# Set kernel
kern=['MATERN_5_2']
kern_ops = []

kern_var=['MATERN_5_2']
kern_var_ops = []

# Set output warp
ow_model=['nat_log', 'meanstd']

ow_noise=['nat_log', 'meanstd']

# Input warp 
iw = True

# GP file (from previous training)
# gp_file = "2D_test.pkl"
gp_file = None
# set class
gp =  GP_hetscedat_class(X, y, y_var, kern, kern_ops,  kern_var, kern_var_ops, iw, ow_model, ow_noise)

# set test and train split
gp.set_test_train(train_mean=0.7, train_noise=0.7)

######################################################
# Noise Model
######################################################

noise_theta_file = None

if noise_theta_file is not None:
    theta = np.load(noise_theta_file)
    gp.set_theta(theta, model='noise')
else:
    # Optimise noise gp
     gp.optimise_gp(model='noise', solver='opt', n_restarts=5)

print(gp.theta_noise)

# Test train plots
gp.test_train_plots(model='noise', fname='noise_plot.png')


######################################################
# Main Model
######################################################
theta_file = None

if theta_file is not None:
    theta = np.load(theta_file)
    gp.set_theta(theta, model='mean')
else:
    # Optimise noise gp
     gp.optimise_gp(model='mean', solver='opt', n_restarts=5)

print(gp.theta)

# Test train plots
gp.test_train_plots(model='mean', fname='mean_plot.png')
