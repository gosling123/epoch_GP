import numpy as np
import scipy.stats 
from GP_model import kernels
import warnings
warnings.filterwarnings('ignore')
import sys
from GP_model.transforms import zero_one_scale, kumaraswamy, output_warp
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import multiprocessing


# Kernel Checker
def check_kernels(kern, kern_ops, kern_list, ops_list):
    if isinstance(kern, list) and all(isinstance(k, str) and k in kern_list for k in kern):
        if len(kern_ops) != len(kern) - 1:
            sys.exit('(ERROR!) : Kernel operations must have length len(kern)-1, as only n-1 operations are required for n combined kernels')
    else:
        sys.exit(f'(ERROR!) : Kernel name not valid, accepted labels are {kern_list}')
    if not all(op in ops_list for op in kern_ops):
        sys.exit(f'(ERROR!) : Kernel operation not recognised, allowed operation labels are {ops_list}')
        

# Class for 1D GP
class GP_class:

    def __init__(self, X, y, y_var, kern=['RBF'], kern_ops=[],\
                 kern_var=['EXP'], kern_var_ops=[], ow_noise=['nat_log'], ow_model=['nat_log']):
        
        # Set inputs and outputs
        self.X = X
        self.n_inputs = self.X.shape[-1]
        self.n_iw_params = int(2*self.n_inputs)
        self.y = y
        self.y_var = y_var

        # set allowed kernel lists
        self.kern_list = ['EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF', 'RAT_QUAD']
        self.ops_list = ['*', '+']

        # set kernel description
        self.kern = kern # mean model
        self.kern_ops = kern_ops
        self.kern_var = kern_var # variance model
        self.kern_var_ops = kern_var_ops
        
        # check kernels
        check_kernels(self.kern_var, self.kern_var_ops, self.kern_list, self.ops_list)
        print('y_var model kernel description accepted')
        check_kernels(self.kern, self.kern_ops, self.kern_list, self.ops_list)
        print('y model kernel description accepted')

        # scale inputs between 0 and 1
        self.sc = zero_one_scale(eps=0.01, x_max=np.max(self.X), x_min=np.min(self.X))
        self.X_sc = self.sc.transform(self.X)

        # Output warping
        self.ow_noise = ow_noise
        self.ow_model = ow_model
       
    #############################################################
    # GP model
    #############################################################

    # Set test train
    def set_test_train(self, train_noise, train_mean):
        # Extract test/train values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_sc , self.y, test_size=1.0-train_mean)
        self.X_var_train, self.X_var_test, self.y_var_train, self.y_var_test = train_test_split(self.X_sc , self.y_var, test_size=1.0-train_noise)


    # Guess priors for parameters
    def set_priors(self, model):
        
        if model == 'noise':
            # Guess for Gaussian noise
            noise_prior = scipy.stats.halfnorm(loc=1e-6, scale=5e-3)
            ow = self.ow_noise
            kern = self.kern_var
        elif model == 'mean':
            noise_prior = scipy.stats.halfnorm(loc=1e-6, scale=1e-4)
            ow = self.ow_model
            kern = self.kern

        # Guess of length-scale prior
        l_prior = scipy.stats.invgamma(a=2, scale=1)
        
        # Guess of kernel variance prior
        kern_var_prior = scipy.stats.invgamma(a=2, scale=1)
        
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
        
        # IW param locations
        if model == 'noise':
            self.iw_noise_idx = len(priors)
        elif model == 'mean':
            self.iw_idx = len(priors)

        # Set output warp params
        if ow is not None:
            for i in range(len(ow)):
                if ow[i] in ('affine', 'sinharcsinh'):
                    priors.extend([ow_a_prior, ow_b_prior])
                elif ow[i] in ('boxcox', 'unit_var'):
                    priors.append(ow_a_prior)
                elif ow[i] in ('zero_mean'):
                    priors.append(ow_b_prior)
                elif ow[i] in ('nat_log', 'meanstd'):
                    continue
        
        if model == 'noise':  
            self.ow_noise_idx = len(priors)
        elif model == 'mean':
            self.ow_idx = len(priors)

        # Gaussian Noise
        priors.append(noise_prior)

        # Location of hypers idx
        if model == 'noise':
            self.hypers_noise_idx = len(priors)
        elif model == 'mean':
            self.hypers_idx = len(priors)
        # Set kernel parameters
        for i in range(len(kern)): 
            if kern[i] in ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF'):
                priors.extend([kern_var_prior, l_prior])
            elif kern[i] == 'RAT_QUAD':
                priors.extend([kern_var_prior, l_prior, alpha_prior])

        return priors
    
    # input warp
    def input_warp(self, X, theta, model):
        # Set parameters
        if model == 'noise':
            a, b = theta[0:self.iw_noise_idx]
        elif model == 'mean':
            a, b = theta[0:self.iw_idx]
        # Warp with kumaraswamy distribution
        iw = kumaraswamy(a, b)
        return iw.transform(X)
    
    # Noise output warp
    def output_warp(self, y, theta, model):
        
        # Set class for noise model parameters
        if model == 'noise':
            ow_params = theta[self.iw_noise_idx:self.ow_noise_idx]
            self.owc_noise = output_warp(warpings=self.ow_noise, params=ow_params, y=y)
            y_warped = self.owc_noise.transform(y)
            jac = self.owc_noise.Jacobian(y)
        
        # Set class for mean model parameters
        elif model == 'mean':
            ow_params = theta[self.iw_idx:self.ow_idx]
            self.owc = output_warp(warpings=self.ow_model, params=ow_params, y=y)
            y_warped = self.owc.transform(y)
            jac = self.owc.Jacobian(y)
        
        return y_warped, jac

    # Get noise kernel
    def set_kernel(self, X_a, X_b, theta, model):    

        # Hyper-paramters stored after warp and noise params
        if model == 'noise':
            hypers = theta[self.hypers_noise_idx:]
            kern = self.kern_var
            kern_ops = self.kern_var_ops        
        elif model == 'mean':
            hypers = theta[self.hypers_idx:]
            kern = self.kern
            kern_ops = self.kern_ops
        
        # Warp inputs
        X_a = self.input_warp(X_a, theta, model)
        X_b = self.input_warp(X_b, theta, model)
        
        # Make kernel
        K = kernels.make_kernel(kern, kern_ops, X_a, X_b, hypers)
        return K
                    
    # Get noise model weights and kernel
    def update_gp(self, theta, model):

        if model == 'noise':
            # Noise
            self.sigma_n = theta[self.ow_noise_idx:self.hypers_noise_idx]
            # Output warp
            self.y_warp, self.jac = self.output_warp(self.y_var_train, theta, model)
            # Input 
            X = self.X_var_train

        elif model == 'mean':
            # Noise
            self.sigma_n = theta[self.ow_idx:self.hypers_idx]
            # Output warp
            self.y_warp, self.jac = self.output_warp(self.y_train, theta, model)
            # Input 
            X = self.X_train
            
        # Kernel
        self.K = self.set_kernel(X, X, theta, model)
        self.K += np.eye(len(X)) * self.sigma_n
        # Weights
        self.L = np.linalg.cholesky(self.K)
        self.weights = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_warp))
        
    # Predictive posterior
    def posterior_predict(self, X_star, model, scale=False, get_var=False):
      
        # Re-scale
        if scale:
            X_star = self.sc.transform(X_star)
        
        # Extract posterior prediction
        mu, var = self.__posterior_predict(X_star, model)
        # Get additional noise prediction
        if model == 'mean':
            mu_noise, var_noise = self.__posterior_predict(X_star, model='noise')

        # Return variance
        if get_var:
            if model == 'mean':
                return mu, var, mu_noise
            elif model == 'noise':
                return mu, var
        else:
            return mu
        
    def __posterior_predict(self, X_star, model):
        
        # Set required GP variables
        if model == 'noise':
            theta = self.theta_noise
            self.update_gp(theta, model)
            X = self.X_var_train
            ow = self.ow_noise
        elif model == 'mean':
            theta = self.theta
            self.update_gp(theta, model)
            X = self.X_train
            ow = self.ow_model

        # Posterior Mean
        K_star = self.set_kernel(X_star, X_star, theta, model)
        k_star = self.set_kernel(X, X_star, theta, model)
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
        if ow is not None:
            mu, var = self.gauss_hermite_quad(mu, var, model)

        return mu, var

        
    ###########################################################################
    # Optimise by gradient descent
    ###########################################################################

    def optimise_ll(self):
        # Extract log likelihood to minimise/maximise
        def get_log_likelihood(theta):
            self.update_gp(theta, self.model)
            sign, logdet = np.linalg.slogdet(self.K)
            n = len(self.K.diagonal())
            log_L = -0.5*np.dot(self.y_warp.T, self.weights) - 0.5*logdet - 0.5*n*np.log(2*np.pi) + np.sum(np.log(self.jac))
            return -1.0*log_L
        return get_log_likelihood

    # Optimisation routine using scipy minimize optimize or differential evoloution
    def optimise_gp(self, model, solver='opt', n_restarts=10, method='L-BFGS-B', max_iter=5000, strategy='best1bin', tol=1e-6, save=True):
        
        # Set model 
        self.model = model
        # Priors
        priors = self.set_priors(model)
        bounds = []
        for i in range(len(priors)):
            bounds.append((priors[i].ppf(0.01), priors[i].ppf(0.95)))
 
        
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

        if model == 'noise':
            self.theta_noise = theta
            if save:
                data = {"theta": self.theta_noise,
                        "X": self.X,
                        "y": self.y_var,
                        "y_test": self.y_var_test,
                        "y_train": self.y_var_train,
                        "X_test": self.X_var_test,
                        "X_train": self.X_var_train,
                        "kern": self.kern_var,
                        "kern_ops": self.kern_var_ops,
                        "ow_model": self.ow_noise,
                        "iw_idx": self.iw_noise_idx,
                        "ow_idx": self.ow_noise_idx,
                        "hypers_idx" : self.hypers_noise_idx}
                # Save to a pickle file
                with open("noise_model_1D.pkl", "wb") as file:
                    pickle.dump(data, file)
                print('Saved parameters to noise_model_1D.pkl')
        
        elif model == 'mean':
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
                with open("mean_model_1D.pkl", "wb") as file:
                    pickle.dump(data, file)
                print('Saved parameters to mean_model_1D.pkl')
        


    #############################################################
    # Set parmaters manually
    #############################################################
    
    # Read in paramters dictionary with theta and indicies 
    def read_gp_model(self, file, model):
        
        with open(file, "rb") as file:
            data = pickle.load(file)
        
        if model == 'noise':
            self.theta_noise = data['theta']
            self.X = data["X"]
            self.y_var = data["y"]
            self.y_var_test = data["y_test"]
            self.y_var_train = data["y_train"]
            self.X_var_test = data["X_test"]
            self.X_var_train = data["X_train"]
            self.kern_var = data["kern"]
            self.kern_var_ops = data["kern_ops"]
            self.ow_noise = data["ow_model"]
            self.iw_noise_idx = data['iw_idx']
            self.ow_noise_idx = data['ow_idx']
            self.hypers_noise_idx = data['hypers_idx']
        elif model == 'mean':
            self.theta = data['theta']
            self.X = data["X"]
            self.y = data["y"]
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
        
    
        self.n_inputs = self.X.shape[-1]
        self.n_iw_params = int(2*self.n_inputs)    
        # scale inputs between 0 and 1
        self.sc = zero_one_scale(eps=0.01, x_max=np.max(self.X), x_min=np.min(self.X))
        self.X_sc = self.sc.transform(self.X)
        # Update GP variables
        self.update_gp(data['theta'], model)
  
    
    #############################################################
    # Gauss - Hermite quadrature
    #############################################################
    def gauss_hermite_quad(self, mu, var, model, deg=8):

        # Revert back to normal output space
        x_i, w_i = np.polynomial.hermite.hermgauss(deg)
        for i in range(len(mu)):
            y_i = np.sqrt(2.0*var[i])*x_i + mu[i]
            
            if model == 'noise':
                y_ir = self.owc_noise.inverse(y_i)
            elif model == 'mean':
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
    def test_train_plots(self, model):
        
        fig = plt.figure()

        # Set correct model params
        if model == 'noise':
            y = self.y_var
            X_train = self.X_var_train
            X_test = self.X_var_test
            y_train = self.y_var_train
            y_test = self.y_var_test
            label = r'$\sigma^2_{\mathcal{R}_{SRS}}$'

            # train and test predictions
            y_train_predict, var_train = self.posterior_predict(X_train, model, scale=False, get_var=True) 
            y_test_predict, var_test = self.posterior_predict(X_test, model, scale=False, get_var=True) 
        
        elif model == 'mean':
            y = self.y
            X_train = self.X_train
            X_test = self.X_test
            y_train = self.y_train
            y_test = self.y_test
            label = r'$\mathcal{R}_{SRS}$'

            # train and test predictions
            y_train_predict, var_train, noise_train = self.posterior_predict(X_train, model, scale=False, get_var=True) 
            y_test_predict, var_test, noise_test = self.posterior_predict(X_test, model, scale=False, get_var=True) 
            # var_train += noise_train
            # var_test += noise_test

        # root mean square error
        rmse_train = np.sqrt(np.mean((y_train-y_train_predict)**2))
        rmse_test = np.sqrt(np.mean((y_test-y_test_predict)**2)) 
     
        # standard deviation
        S_ptrain = np.sqrt(var_train)
        S_ptest = np.sqrt(var_test)

        # Plot correlation between values and errors
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
        ax3 = sns.kdeplot(np.array(y), label = 'Target Data', linestyle='dashdot', linewidth = 5, color = 'black')
        ax3 = sns.kdeplot(y_train_predict, label=f'Train', color = 'blue')
        ax3 = sns.kdeplot(y_test_predict, label=f'Test', color = 'orange')
        ax3.legend(frameon=False)
        ax3.set_xlabel(label)

        ax1.scatter(y_train, y_train_predict, label=f'Train (RSME = {np.round(rmse_train, 3)})', color = 'blue')
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k:', label = 'Target')
        ax1.set_xlabel(f'True Value - {label}')
        ax1.set_ylabel(f'Predicted Value - {label}')
        ax2.plot(abs(y_train_predict - y_train), S_ptrain, 'o', label='Train', color = 'blue')
        ax2.plot([0, S_ptrain.max()], [0, S_ptrain.max()], 'k:', label = 'Target')
        ax1.scatter(y_test, y_test_predict, label=f'Test (RSME = {np.round(rmse_test, 3)})', color = 'orange')
        ax1.legend()
        ax2.plot(abs(y_test_predict - y_test), S_ptest, 'o', label='Test', color='orange')
        ax2.plot([0, S_ptest.max()], [0, S_ptest.max()], 'k:', label = 'Target')
        ax2.legend()
        ax2.set_xlabel(f'True Error - {label}')
        ax2.set_ylabel(f'Predicted Error - {label}')
      
        # Make plots spaced out
        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.5, 
                            hspace=0.4)

        plt.savefig(f'{model}_test_train.png')

    
    