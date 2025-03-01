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

# 1D test Data
# X = np.load('./test_data/1D_data/input_mean.npy')
# y = np.load('./test_data/1D_data/output_reflect_mean.npy').flatten()

# print(X.shape())
# 2D test Data
X = np.load('./test_data/2D_data/input_mean.npy')
y = np.load('./test_data/2D_data/output_reflect_mean.npy').flatten()

############################################################################
# Set GP model
############################################################################

# Set kernel
kern=['MATERN_5_2_[1,1]', 'RBF_ISO_[1]' ]
kern_ops = ['*']

# Set output warp
ow_model=['nat_log', 'meanstd']

# GP file (from previous training)
# gp_file = "2D_test.pkl"
gp_file = None
# set class
gp = GP_class(X, y, kern, kern_ops, ow_model)

if gp_file == None:
    # set test and train split
    gp.set_test_train(train_frac=0.7)

######################################################
# GP model train/prediction
######################################################

if gp_file is not None:
    gp.read_gp_model(gp_file)
else:
    # Optimise noise gp
    gp.optimise_gp(solver='opt', n_restarts=6, save=True, fname="2D_test.pkl")
print(gp.theta)

# Test train plots
gp.test_train_plots(fname='2D_test_train.png')
