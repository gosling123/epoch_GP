import numpy as np
import scipy.stats 
from Model import kernels
import warnings
warnings.filterwarnings('ignore')
import sys
from Model.transforms import zero_one_scale, kumaraswamy, output_warp
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle



# # Kernel Checker
# def check_kernels(kern, kern_ops, kern_list, ops_list):
#     if isinstance(kern, list) and all(isinstance(k, str) and k in kern_list for k in kern):
#         if len(kern_ops) != len(kern) - 1:
#             sys.exit('(ERROR!) : Kernel operations must have length len(kern)-1, as only n-1 operations are required for n combined kernels')
#     else:
#         sys.exit(f'(ERROR!) : Kernel name not valid, accepted labels are {kern_list}')
#     if not all(op in ops_list for op in kern_ops):
#         sys.exit(f'(ERROR!) : Kernel operation not recognised, allowed operation labels are {ops_list}')
        

def check_kernels(kern, kern_ops, kern_list, ops_list):

    for k in kern:
        if k in kern_list:
            continue
        elif k.startswith(kern_list) == False:
            sys.exit(f'(ERROR) : Kernel {k} is not defined. Kernels should start with {kern_list}')
    
    if len(kern_ops) != len(kern) - 1:
            sys.exit('(ERROR!) : Kernel operations must have length len(kern)-1, as only n-1 operations are required for n combined kernels')
    if not all(op in ops_list for op in kern_ops):
        sys.exit(f'(ERROR!) : Kernel operation not recognised, allowed operation labels are {ops_list}')        


def contains_all_dims(strings, N):
    # Extract digits from all strings and convert to a set of unique numbers
    digit_set = {int(digit) for s in strings for digit in s if digit.isdigit()}
    
    # Check if all numbers from 1 to N are in the set
    return set(range(1, N + 1)).issubset(digit_set)        


class GP_class:

    def __init__(self, X, y, kern=['RBF'], kern_ops=[], ow_model=['nat_log']):
        
        # Set inputs and outputs
        self.X = X
        self.n_inputs = self.X.shape[-1]
        self.n_iw_params = int(2*self.n_inputs)
        self.y = y

        # set allowed kernel lists
        self.kern_list = ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF', 'RAT_QUAD')
        self.ops_list = ['*', '+']

        # set kernel description
        self.kern = kern # mean model
        self.kern_ops = kern_ops
        
        # check kernels
        check_kernels(self.kern, self.kern_ops, self.kern_list, self.ops_list)
        print('y model kernel description accepted')

        # Check if 'NS' is not present in any of the strings
        if not any('_NS' in s for s in self.kern):
            self.non_sep = False
        else:
            self.non_sep = True

        if self.non_sep and self.n_inputs == 1:
            sys.exit('(ERROR): Cannot have non-seperable kernel for 1 input')

        check_array = []
        for i in range(len(self.kern)):
            check_array.append(kernels.extract_numbers_after_kernel(self.kern[i]))
        self.dim_label_check = kernels.check_dim_nums(check_array)
        
        if self.n_inputs > 1 and self.non_sep == False:
            # Check if assumed seperable in ascending dimesnion order
            if self.dim_label_check == "all_none" and self.non_sep == False:
                if self.n_inputs != len(self.kern):
                    sys.exit('(ERROR) : Please ensure number of kernels (kern) is equal in length to number of input parameters for purely seperable kernels\
                                where no input number is defined')
            
            elif self.dim_label_check == "all_strings":
                all_dim_check = contains_all_dims(check_array, self.n_inputs)
                if all_dim_check == False:
                    sys.exit('(ERROR) : When defining dims for purely seperable case please ensure all dims have been included')
            
            elif self.dim_label_check == "strings_and_none" or self.dim_label_check == "mixed_type":
                sys.exit('(ERROR) purely seperable kernels should all have no numbers after kernel label,\
                          (order is aummed ascending dimesnion), or all have numbers where each kernel marks a given dimension')
   
        elif self.n_inputs > 1 and self.non_sep == True:

            if len(self.kern) == 1 and check_array == "all_strings":
                sys.exit('(ERROR) : Please remove any dim numbers when defining one kernel type for non-seperable inputs (i.e all inputs are non-seperable)')
            
            elif len(self.kern) > 1 and self.dim_label_check == "all_strings":
                all_dim_check = contains_all_dims(check_array, self.n_inputs)
                if all_dim_check == False:
                    sys.exit('(ERROR) : When defining dims for seperable/non-seperable or purely seperable with multiple kernels, please ensure all dims have been included')
            
            elif self.dim_label_check == "strings_and_none" or self.dim_label_check == "mixed_type":
                sys.exit('(ERROR) purely non-seperable using multiple kernels or mixed of seperable and non-seperable should all have numbers where each kernel marks a given dimension')
        
        # scale inputs between 0 and 1
        self.X_sc = np.zeros_like(self.X)
        for i in range(self.n_inputs):
            sc = zero_one_scale(eps=0.01, x_max=np.max(self.X[:,i]), x_min=np.min(self.X[:,i]))
            self.X_sc[:,i] = sc.transform(self.X[:,i])
      
        # Output warping
        self.ow_model = ow_model
       
    #############################################################
    # GP model
    #############################################################

    # Set test train
    def set_test_train(self, train_frac):
        # Extract test/train values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_sc , self.y, test_size=1.0-train_frac)


    # Guess priors for parameters
    def set_priors(self):
        
        # Guess of length-scale prior
        l_prior = scipy.stats.invgamma(a=2, scale=1)
        # l_prior = scipy.stats.lognorm(s=1, scale=1)
        
        # Guess of kernel variance prior
        kern_var_prior = scipy.stats.invgamma(a=2, scale=1)
        # kern_var_prior = scipy.stats.lognorm(s=0.75, scale=np.exp(0.56))
        
        # Guess for rational quadrtatic paramter
        alpha_prior = scipy.stats.invgamma(a=2, scale=1)

        # Prior for input warping parameters a, b
        iw_prior = scipy.stats.lognorm(s=0.25, scale=1)

        # Prior for otput warping parameters a, b
        ow_a_prior = scipy.stats.norm(loc=0, scale=1)
        ow_b_prior = scipy.stats.lognorm(s=0.25, scale=1)
        
        priors = []

        # Input warp
        for i in range(self.n_iw_params):
            priors.append(iw_prior)
        self.iw_idx = len(priors)

        # Set output warp params
        if self.ow_model is not None:
            for i in range(len(self.ow_model)):
                if self.ow_model[i] in ('affine', 'sinharcsinh'):
                    priors.extend([ow_a_prior, ow_b_prior])
                elif self.ow_model[i] in ('boxcox', 'unit_var'):
                    priors.append(ow_a_prior)
                elif self.ow_model[i] in ('zero_mean'):
                    priors.append(ow_b_prior)
                elif self.ow_model[i] in ('nat_log', 'meanstd'):
                    continue
        self.ow_idx = len(priors)

        # Guess for Gaussian nois
        noise_prior = scipy.stats.halfnorm(loc=1e-6, scale=1e-2)
        # Gaussian Noise
        priors.append(noise_prior)

        # Location of hypers idx
        self.hypers_idx = len(priors)

        # Set kernel parameters
        if self.n_inputs == 1:
            if self.dim_label_check == 'all_strings':
                sys.exit('(ERROR) Dimesnion numbers are not required for 1D inputs') 
            
            for i in range(len(self.kern)): 
                if self.kern[i] in ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF'):
                    priors.extend([kern_var_prior, l_prior])
                elif self.kern[i] == 'RAT_QUAD':
                    priors.extend([kern_var_prior, l_prior, alpha_prior])
        
        elif len(self.kern) == 1 and self.non_sep:
            # All non-seperable
            label = self.kern[0].split("_NS")[0]
            # Number of vals to compute non-seperable cov matrix
            nvals = int(0.5*self.n_inputs*(self.n_inputs+1))
            # Random Cholesky factors assumed to be related to lengthscale
            if label in ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF'):
                priors.append(kern_var_prior)
                for i in range(nvals):
                    priors.append(l_prior)
            elif label == 'RAT_QUAD':
                priors.append(kern_var_prior)
                for i in range(nvals):
                    priors.append(l_prior)
                priors.append(alpha_prior)
            
        # Mixed (seperable and or non-seperable)
        else:
            for i in range(len(self.kern)):
                check = self.kern[i].split("_NS")[0] if "_NS" in self.kern[i] else None
                
                # Seperable
                if check == None:
                    if self.dim_label_check == 'all_none':
                        label = self.kern[i] 
                    else:
                        label = kernels.get_kernel_label(self.kern[i])
                     
                    if label in ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF'):
                        priors.extend([kern_var_prior, l_prior])
                    elif label == 'RAT_QUAD':
                        priors.extend([kern_var_prior, l_prior, alpha_prior])
                    
                # Non-seperable
                else:
                    dim = kernels.extract_numbers_after_kernel(self.kern[i])

                    # Number of vals to compute non-seperable cov matrix
                    nvals = int(0.5*len(dim)*(len(dim)+1))
                    # Which kernel
                    label = self.kern[i].split("_NS")[0]
                    # Random Cholesky factors assumed to be related to lengthscale
                    if label in ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF'):
                        priors.append(kern_var_prior)
                        for i in range(nvals):
                            priors.append(l_prior)
                    elif label == 'RAT_QUAD':
                        priors.append(kern_var_prior)
                        for i in range(nvals):
                            priors.append(l_prior)
                        priors.append(alpha_prior)

        return priors
    
    # input warp
    def input_warp(self, X, theta):
        # Parameters
        iw_params = theta[0:self.iw_idx]
        # Warp with kumaraswamy distribution
        X_iw = np.zeros_like(X)
        idx = 0
        for i in range(self.n_inputs):
            iw = kumaraswamy(a=iw_params[idx], b=iw_params[idx+1])
            X_iw[:,i] = iw.transform(X[:,i])
            idx += 2
        return X_iw
    
    # Noise output warp
    def output_warp(self, y, theta):
        # Set class for output warping
        ow_params = theta[self.iw_idx:self.ow_idx]
        self.owc = output_warp(warpings=self.ow_model, params=ow_params, y=y)
        y_warped = self.owc.transform(y)
        jac = self.owc.Jacobian(y)
        return y_warped, jac

    # Get noise kernel
    def set_kernel(self, X_a, X_b, theta):    
        # Hyper-parameters
        hypers = theta[self.hypers_idx:]    
        # Warp inputs
        X_a = self.input_warp(X_a, theta)
        X_b = self.input_warp(X_b, theta)

        # Make kernel
        if self.n_inputs == 1 and self.non_sep == False:
            K = kernels.make_kernel(self.kern, self.kern_ops, X_a, X_b, hypers)
        elif self.n_inputs > 1 and self.non_sep == False:
            K = kernels.make_kernel_nD(self.kern, self.kern_ops, X_a, X_b, hypers)
        elif self.non_sep == True:
            K = kernels.make_kernel_NS(self.kern, self.kern_ops, X_a, X_b, hypers)
        return K
                    
    # Get noise model weights and kernel
    def update_gp(self, theta):

        # Noise
        self.sigma_n = theta[self.ow_idx:self.hypers_idx]
        # Output warp
        self.y_warp, self.jac = self.output_warp(self.y_train, theta)
            
        # Kernel
        self.K = self.set_kernel(self.X_train, self.X_train, theta)
        self.K += np.eye(len(self.X_train)) * self.sigma_n

        try:
            # Weights
            self.L = np.linalg.cholesky(self.K)
        except np.linalg.LinAlgError:
            sys.exit('Defined kernel is NOT positive defnite, please make another kernel')
        self.weights = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_warp))
        
    # Predictive posterior
    def posterior_predict(self, X_star, scale=False, get_var=False):
        
        # Re-scale
        if scale:
            X_sc = np.zeros_like(X_star)
            for i in range(self.n_inputs):
                sc = zero_one_scale(eps=0.01, x_max=np.max(self.X[:,i]), x_min=np.min(self.X[:,i]))
                X_sc[:,i] = sc.transform(X_star[:,i])
            X_star = X_sc
        
        # Extract posterior prediction
        mu, var = self.__posterior_predict(X_star)
        
        # Return variance
        if get_var:
            return mu, var
        else:
            return mu
        
    def __posterior_predict(self, X_star):
        
        # Set required GP variables
        self.update_gp(self.theta)
   
        # Posterior Mean
        K_star = self.set_kernel(X_star, X_star, self.theta)
        k_star = self.set_kernel(self.X_train, X_star, self.theta)
        mu = np.dot(k_star.T, self.weights)

        # Variance
        v = np.linalg.solve(self.L, k_star)
        V_star = K_star - np.dot(v.T, v)
        # Epistemic 
        var_epi = np.diag(V_star)
        # Noise
        var_noise = np.ones(len(var_epi)) * self.sigma_n
        var = var_epi + var_noise

        # If warped revert
        if self.ow_model is not None:
            mu, var = self.gauss_hermite_quad(mu, var)

        return mu, var

        
    ###########################################################################
    # Optimise by gradient descent
    ###########################################################################

    def optimise_ll(self):
        # Extract log likelihood to minimise/maximise
        def get_log_likelihood(theta):
            self.update_gp(theta)
            sign, logdet = np.linalg.slogdet(self.K)
            n = len(self.K.diagonal())
            log_L = -0.5*np.dot(self.y_warp.T, self.weights) - 0.5*logdet - 0.5*n*np.log(2*np.pi) + np.sum(np.log(self.jac))
            return -1.0*log_L
        return get_log_likelihood

    # Optimisation routine using scipy minimize optimize or differential evoloution
    def optimise_gp(self, solver='opt', n_restarts=10, method='L-BFGS-B', max_iter=5000, strategy='best1bin', tol=1e-6, save=True, fname='mean_model_nD.pkl'):
        
        # Priors
        priors = self.set_priors()
        bounds = []
        for i in range(len(priors)):
            bounds.append((priors[i].ppf(0.01), priors[i].ppf(0.99)))

        if solver == 'diff_evo':
            # Set progress to estimate maximum time
            progress_bar = tqdm(total=max_iter, desc="Differential Evolution")
            # Custom callback to update the progress bar
            def callback(xk, convergence):
                progress_bar.update(1)
                if progress_bar.n >= max_iter:
                    progress_bar.close()
        
            # Perform global minimisation
            res =  differential_evolution(self.optimise_ll(), bounds, strategy=strategy, tol=tol, maxiter=max_iter, callback=callback)

            theta = res.x
            # Close the progress bar after completion
            progress_bar.close()
        
        elif solver == 'opt':
            # Perform gradient optimisation with multiple restarts
            for n in range(n_restarts):
                # Random intial start taken from prior
                initial = np.zeros(len(priors))
                for i in range(len(priors)):
                    initial[i] = priors[i].rvs(1)
        
                # Perform minimisation
                res = minimize(self.optimise_ll(), initial,
                               bounds=tuple(bounds), method=method)
            
                print(f'restart {n} = {res.fun}')
            
                # Accept first value
                if n == 0:
                    log_L = res.fun
                    theta = res.x
                # Only accept lowest value
                else:
                    if res.fun < log_L:
                        theta = res.x
                        log_L = res.fun

        # Store optimised parameters
        self.theta = theta
        if save:
            data = {"theta": self.theta,
                    "X": self.X,
                    "y": self.y,
                    "y_test": self.y_test,
                    "y_train": self.y_train,
                    "X_test": self.X_test,
                    "X_train": self.X_train,
                    "kern": self.kern,
                    "kern_ops": self.kern_ops,
                    "ow_model": self.ow_model,
                    "iw_idx": self.iw_idx,
                    "ow_idx": self.ow_idx,
                    "hypers_idx" : self.hypers_idx}
            # Save to a pickle file
            with open(fname, "wb") as file:
                pickle.dump(data, file)
            print(f'Saved parameters to {fname}')
        

    #############################################################
    # Set parmaters manually
    #############################################################
    
    # Read in paramters dictionary with theta and indicies 
    def read_gp_model(self, file):
        
        with open(file, "rb") as file:
            data = pickle.load(file)
        
        self.X = data["X"]
        self.y = data["y"]
        self.n_inputs = self.X.shape[-1]
        self.n_iw_params = int(2*self.n_inputs)
        
        # scale inputs between 0 and 1
        self.X_sc = np.zeros_like(self.X)
        for i in range(self.n_inputs):
            sc = zero_one_scale(eps=0.01, x_max=np.max(self.X[:,i]), x_min=np.min(self.X[:,i]))
            self.X_sc[:,i] = sc.transform(self.X[:,i])

        self.theta = data['theta']
        self.y_test = data["y_test"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.X_train = data["X_train"]
        self.kern = data["kern"]
        self.kern_ops = data["kern_ops"]
        self.ow_model = data["ow_model"]
        self.iw_idx = data['iw_idx']
        self.ow_idx = data['ow_idx']
        self.hypers_idx = data['hypers_idx']
        # Update GP variables
        self.update_gp(data['theta'])
  
    
    #############################################################
    # Gauss - Hermite quadrature
    #############################################################
    def gauss_hermite_quad(self, mu, var, deg=8):

        # Revert back to normal output space
        x_i, w_i = np.polynomial.hermite.hermgauss(deg)
        for i in range(len(mu)):
            y_i = np.sqrt(2.0*var[i])*x_i + mu[i]

            # invert to pre warped space
            y_ir = self.owc.inverse(y_i)

            # New mean
            mu[i] = 1.0/np.sqrt(np.pi) * np.sum(w_i*y_ir)

            # New Variance
            y_ir2 = np.power(y_ir, 2)
            var[i] = 1.0/np.sqrt(np.pi) * np.sum(w_i*y_ir2) - mu[i]**2
        
        return mu, var

   
    #############################################################
    # Plotters
    #############################################################
    # Test train plots
    def test_train_plots(self, fname=f'test_train.png'):
        
        fig = plt.figure()

        # train and test predictions
        y_train_predict, var_train = self.posterior_predict(self.X_train, scale=False, get_var=True) 
        y_test_predict, var_test = self.posterior_predict(self.X_test, scale=False, get_var=True) 
        

        # root mean square error
        rmse_train = np.sqrt(np.mean((self.y_train-y_train_predict)**2))
        rmse_test = np.sqrt(np.mean((self.y_test-y_test_predict)**2)) 
     
        # standard deviation
        S_ptrain = np.sqrt(var_train)
        S_ptest = np.sqrt(var_test)

        # Plot correlation between values and errors
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
        ax3 = sns.kdeplot(np.array(self.y), label = 'Target Data', linestyle='dashdot', linewidth = 5, color = 'black')
        ax3 = sns.kdeplot(y_train_predict, label=f'Train', color = 'blue')
        ax3 = sns.kdeplot(y_test_predict, label=f'Test', color = 'orange')
        ax3.legend(frameon=False)
        ax3.set_xlabel(r'Output Values')

        ax1.scatter(self.y_train, y_train_predict, label=f'Train (RSME = {np.round(rmse_train, 3)})', color = 'blue')
        ax1.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k:', label = 'Target')
        ax1.set_xlabel(r'True Value')
        ax1.set_ylabel(r'Predicted Value')
        ax2.plot(abs(y_train_predict - self.y_train), S_ptrain, 'o', label='Train', color = 'blue')
        ax2.plot([0, S_ptrain.max()], [0, S_ptrain.max()], 'k:', label = 'Target')
        ax1.scatter(self.y_test, y_test_predict, label=f'Test (RSME = {np.round(rmse_test, 3)})', color = 'orange')
        ax1.legend()
        ax2.plot(abs(y_test_predict - self.y_test), S_ptest, 'o', label='Test', color='orange')
        ax2.plot([0, S_ptest.max()], [0, S_ptest.max()], 'k:', label = 'Target')
        ax2.legend()
        ax2.set_xlabel(r'True Error')
        ax2.set_ylabel(r'Predicted Error')
        ax1.set_ylim(self.y.min(), self.y.max())
        # Make plots spaced out
        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.5, 
                            hspace=0.4)

        plt.savefig(fname, bbox_inches='tight')

    
