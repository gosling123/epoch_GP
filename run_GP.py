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
X = np.load('./test_data/4D_data/Inputs_4D_model_1_2000.npy')
y = np.abs(np.load('./test_data/4D_data/R_bsrs_4D_model_1_2000.npy'))
y+=1e-6

############################################################################
# Set GP model
############################################################################

# Set kernel
kern=['MATERN_5_2_[1]', 'MATERN_5_2_[2]', 'MATERN_5_2_[3]', 'MATERN_5_2_[4]']
kern_ops = "k_1 * (k_2 + k_3 + k_4)"

# Set output warp
# ow_model=['nat_log', 'unit_var', 'sinharcsinh', 'affine', 'sinharcsinh', 'meanstd']
ow_model=['nat_log', 'meanstd']

# GP file (from previous training)
gp_file = None

# set classs
gp = GP_class(X, y, kern, kern_ops, ow_model)

if gp_file == None:
    # set test and train split
    gp.set_test_train(train_frac=0.1)

######################################################
# GP model train/prediction
######################################################

if gp_file is not None:
    gp.read_gp_model(gp_file)
else:
    # Optimise noise gp
    gp.optimise_gp(solver='opt', n_restarts=10, save=True, fname="R_bsrs_model.pkl")
print(gp.theta)

# Test train plots
gp.test_train_plots()
