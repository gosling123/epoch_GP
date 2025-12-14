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


def check_kernels(kern, kern_ops, kern_list, ops_list):
    
    """
    Check that kernel description is allowes

    Parameters:
    kern : list of str
        Kernel label.
    kern_ops : str
        String defining overll kernel e.g "k_1 * (k_2 + k_3) + k_4"
    kern_list : list of str
        List of allowed kernels.
    ops_list : list of str
        List of allowed kernel operations ('+' or '*').
        
    """

    # --------------------------------------------------
    # 1. Validate kernel labels provided in `kern`
    #    Ensures each kernel name starts with an allowed prefix
    # --------------------------------------------------
    for k in kern:
        if k in kern_list:
            continue
        if not k.startswith(kern_list):
            sys.exit(
                f"(ERROR): Kernel {k} is not defined. "
                f"Kernels should start with {kern_list}"
            )

    # --------------------------------------------------
    # 2. Parse `kern_ops` character-by-character
    #    - Extract k_<integer> tokens
    #    - Validate that only allowed symbols appear
    # --------------------------------------------------
    found = set()                 # stores kernel indices found in kern_ops
    i = 0                         # current position in the string

    allowed_ops = set(ops_list)   # e.g. {'+', '*'}
    allowed_single = allowed_ops | {'(', ')', ' '}  # valid single-char tokens

    while i < len(kern_ops):
        c = kern_ops[i]

        # ---- Handle kernel references: k_<integer> ----
        if c == 'k':
            # Must be followed by an underscore
            if i + 1 >= len(kern_ops) or kern_ops[i + 1] != '_':
                sys.exit("(ERROR): Invalid kernel token in kern_ops")

            j = i + 2  # position after 'k_'

            # At least one digit must follow k_
            if j >= len(kern_ops) or not kern_ops[j].isdigit():
                sys.exit("(ERROR): Kernel index must be an integer")

            # Read all consecutive digits to support k_10, k_123, etc.
            num = 0
            while j < len(kern_ops) and kern_ops[j].isdigit():
                num = num * 10 + int(kern_ops[j])
                j += 1

            # Record the kernel index we just parsed
            found.add(num)

            # Advance index past the entire k_<number> token
            i = j
            continue

        # ---- Handle allowed single-character tokens ----
        # Operators, parentheses, and whitespace
        if c in allowed_single:
            i += 1
            continue

        # ---- Anything else is illegal ----
        sys.exit(f"(ERROR): Illegal character '{c}' in kern_ops")

    # --------------------------------------------------
    # 3. Check kernel coverage
    #    Ensure k_1 ... k_n all appear at least once
    # --------------------------------------------------
    required = set(range(1, len(kern) + 1))  # expected kernel indices
    missing = required - found               # kernels never referenced

    if missing:
        sys.exit(
            f"(ERROR): Missing kernel references in kern_ops: "
            f"{', '.join(f'k_{i}' for i in sorted(missing))}"
        )

    # All checks passed
    return True

class GP_hetscedat_class:

    def __init__(self, X, y, y_var, kern=['RBF'], kern_ops="k_1",  kern_var=['EXP'], kern_var_ops="k_1",\
                  iw=True, ow_model=['nat_log'], ow_noise=['nat_log']):

        """
        Initialise Gaussian Process (heteroscedatic) class

        Parameters:
        X : ndarray
            Input array for training.
        y : ndarray
            Output array for training (Only handles single QOI output).
        y_var : ndarray
            Output array for training input noise variance
        kern : list of str
            List of kernels to use.
        kern_ops : str
            String defining overll kernel e.g "k_1 * (k_2 + k_3) + k_4".
        kern_var : list of str
            List of kernels to use (for variance model).
        kern_ops : str
            String defining overll kernel e.g "k_1 * (k_2 + k_3) + k_4". (Variance model)
        iw : logical flag
            Flag to turn input warping on or off
        ow_model : list of str
            List of output warpings to perform.
        ow_noise : list of str
            List of output warpings to perform (for variance model).
        
        """
        
        # Set inputs and outputs
        self.X = X

        if self.X.ndim == 1:
            self.X = self.X[:, None]  # Reshape (n,) to (n,1)

        self.n_inputs = self.X.shape[-1]
        if iw :
            self.n_iw_params = int(2*self.n_inputs)
        self.y = y
        self.y_var = y_var

        # set allowed kernel lists
        self.kern_list = ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF', 'RAT_QUAD')
        self.ops_list = ['*', '+']

        # set kernel description
        self.kern = kern # mean model
        self.kern_ops = kern_ops
        self.kern_var = kern_var # variance model
        self.kern_var_ops = kern_var_ops
        
        # check kernels
        check_kernels(self.kern, self.kern_ops, self.kern_list, self.ops_list)
        print('y model kernel description accepted')
        check_kernels(self.kern_var, self.kern_var_ops, self.kern_list, self.ops_list)
        print('y_var model kernel description accepted')

        for kerns in (self.kern, self.kern_var):
            # Check if 'NS' is not present in any of the strings
            if not any('_NS' in s for s in kerns):
                NS = False
            else:
                NS = True

            # Check if 'ARD' is not present in any of the strings
            if not any('_ARD' in s for s in kerns):
                ARD = False
            else:
                ARD = True

            # Check if '_ISO' is not present in any of the strings
            if not any('_ISO' in s for s in kerns):
                ISO = False
            else:
                ISO = True
  
            check_array = []
            for i in range(len(kerns)):
                val = kernels.extract_numbers_after_kernel(kerns[i])
                if val == None:
                    check_array.append(val)
                elif isinstance(val, list) and all(isinstance(x, int) for x in val):
                    check_array.extend(val)
                else:
                    sys.exit('(ERROR) Issue with naming. Please use the following form e.g RBF, RBF_ISO_[1,2..], RBF_NS, RBF_NS_[1, 2..], RBF_ARD, RBF_ARD_[1,2..]')
            
            self.dim_label_check = kernels.check_dim_nums(check_array)
        
            check_dims = kernels.extract_numbers(kerns)
            if all(isinstance(x, int) and 0 <= x <= self.n_inputs for x in check_dims) == False:
                sys.exit('(ERROR) All specified dimensions should be an integer between 1 and N_inputs')

            # 1D case check
            if self.n_inputs == 1:
                # Checks for various kernel descriptions
                if (NS or ISO or ARD):
                    sys.exit('(ERROR): Cannot have non-seperable or ARD kernel for 1 input. In 1D already isotropic so drop _ISO flag')
                if self.dim_label_check != 'all_none':
                    sys.exit('(ERROR): 1D so no need to specify dimensions. Remove numbers list from label i.e RBF not RBF_[1,2]')
        
            # nD case
            else:
                if self.dim_label_check == "all_integers" and set(range(1, self.n_inputs + 1)).issubset(set(check_dims)) == False:
                    sys.exit('(ERROR) All input dimensions should be used in kernel description')
                
        
        # scale inputs between 0 and 1
        self.X_sc = np.zeros_like(self.X)
        for i in range(self.n_inputs):
            sc = zero_one_scale(eps=0.01, x_max=np.max(self.X[:,i]), x_min=np.min(self.X[:,i]))
            self.X_sc[:,i] = sc.transform(self.X[:,i])

        # Check to input warp
        self.iw = iw

        if ow_model is None or (isinstance(ow_model, list) and all(isinstance(item, str) and len(item) >= 1 for item in ow_model)):
            # Output warping
            self.ow_model = ow_model
        else:
            sys.exit('(ERROR) : ow_model must be set to None for no warping, or a list of allowed opeartions given by the strngs affine,nat_log, boxcox, sinharcsinh, meanstd,\
                   zero_mean, unit_var')
        
        if ow_noise is None or (isinstance(ow_noise, list) and all(isinstance(item, str) and len(item) >= 1 for item in ow_noise)):
            # Output warping
            self.ow_noise = ow_noise
        else:
            sys.exit('(ERROR) : ow_noise must be set to None for no warping, or a list of allowed opeartions given by the strngs affine,nat_log, boxcox, sinharcsinh, meanstd,\
                   zero_mean, unit_var')

    def rescale_X(self, X_sc):
        """
        Return re-scaled version from 0, 1 scale
        
        Parameters:
        X_sc : ndarray
            Scaled inputs
        
        Returns:
        ndarray
            Inputs in original space.
        
        """
        # re-scale inputs
        X = np.zeros_like(X_sc)
        for i in range(self.n_inputs):
            sc = zero_one_scale(eps=0.01, x_max=np.max(self.X[:,i]), x_min=np.min(self.X[:,i]))
            X[:,i] = sc.inverse(X_sc[:,i])
        return X
           
    #############################################################
    # GP model
    #############################################################

    # Set test train
    def set_test_train(self, train_mean, train_noise):
        """
        Set test train split for given fraction
        
        Parameters:
        train_mean : ndarray
            Percentage of data to train mean data on
        train_noise : ndarray
            Percentage of data to train noise data on
        
        """
        # Extract test/train values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_sc , self.y, test_size=1.0-train_mean)
        self.X_var_train, self.X_var_test, self.y_var_train, self.y_var_test = train_test_split(self.X_sc , self.y_var, test_size=1.0-train_noise)


    # Guess priors for parameters
    def set_priors(self, model):
        """
        Set priors

        model : str
            String flag to set GP on mean or variance (noise) model data

        Returns:
        ndarray
            The list of parameter priors.

        """ 

        if model == 'noise':
            # Guess for Gaussian nois
            noise_prior = scipy.stats.halfnorm(loc=1e-6, scale=1e-2)
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
        param_labels = []

        if self.iw:
            # Input warp
            for i in range(self.n_iw_params):
                priors.append(iw_prior)
                if (i+1) % 2 == 1:  # Odd number
                    param_labels.append('iw_a')
                else:
                    param_labels.append('iw_b')
            if model == 'noise':
                self.iw_noise_idx = len(priors)
            elif model == 'mean':
                self.iw_idx = len(priors)
        else:
            if model == 'noise':
                self.iw_noise_idx = 0
            elif model == 'mean':
                self.iw_idx = 0

        # Set output warp params
        if ow is not None:
            for i in range(len(ow)):
                if ow[i] in ('affine', 'sinharcsinh'):
                    priors.extend([ow_a_prior, ow_b_prior])
                    param_labels.extend([f'{ow[i]}_a', f'{ow[i]}_b'])
                elif ow[i] in ('boxcox', 'unit_var'):
                    priors.append(ow_a_prior)
                    param_labels.append(f'{ow[i]}_a')
                elif ow[i] in ('zero_mean'):
                    priors.append(ow_b_prior)
                    param_labels.append(f'{ow[i]}_b')
                elif ow[i] in ('nat_log', 'meanstd'):
                    continue
            if model == 'noise':
                self.ow_noise_idx = len(priors)
            elif model == 'mean':
                self.ow_idx = len(priors)
        else:
            if model == 'noise':
                self.ow_noise_idx = self.iw_noise_idx
            elif model == 'mean':
                self.ow_idx = self.iw_idx

        # Gaussian Noise
        priors.append(noise_prior)
        param_labels.append('sigma_n')

        # Location of hypers idx
        if model == 'noise':
            self.hypers_noise_idx = len(priors)
        elif model == 'mean':
            self.hypers_idx = len(priors)

        # Set kernel parameters
        if self.n_inputs == 1:      
            for i in range(len(kern)): 
                if kern[i] in ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF'):
                    priors.extend([kern_var_prior, l_prior])
                    param_labels.extend([f'{kern[i]}_var', f'{kern[i]}_l'])
                elif kern[i] == 'RAT_QUAD':
                    priors.extend([kern_var_prior, l_prior, alpha_prior])
                    param_labels.extend([f'{kern[i]}_var', f'{kern[i]}_l', f'{kern[i]}_alpha'])
        else:
            for i in range(len(kern)):
                k_type = kernels.kernel_type(kern[i])
                label = kernels.get_kernel_label(kern[i])
                dims = kernels.extract_numbers_after_kernel(kern[i])

                if k_type in {"_ISO",  None}:
                    # Isotropic (all dimensions, or given) or seperable with given dimensions
                    if label in ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF'):
                        priors.extend([kern_var_prior, l_prior])
                        param_labels.extend([f'{kern[i]}_var', f'{kern[i]}_l'])
                    elif label == 'RAT_QUAD':
                        priors.extend([kern_var_prior, l_prior, alpha_prior])
                        param_labels.extend([f'{kern[i]}_var', f'{kern[i]}_l', f'{kern[i]}_alpha'])

                elif k_type in {"_ARD", "_NS"}:
                    # One variance but seperate length-scales

                    if dims == None:
                        if k_type == "_ARD":
                            nvals = self.n_inputs
                            string = "l"
                        elif k_type == "_NS":
                            nvals = int(0.5*self.n_inputs*(self.n_inputs+1))
                            string = "lambda_inv"
                    else:
                        if k_type == "_ARD":
                            nvals = len(dims)
                        elif k_type == "_NS":
                            nvals = int(0.5*len(dims)*(len(dims)+1))

                    if label in ('EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF'):
                        priors.append(kern_var_prior)
                        param_labels.append(f'{kern[i]}_var')
                        for j in range(nvals):
                            priors.append(l_prior)
                            param_labels.append(f'{kern[i]}_{string}_{j+1}')
                    elif label == 'RAT_QUAD':
                        priors.append(kern_var_prior)
                        param_labels.append(f'{kern[i]}_var')
                        for j in range(nvals):
                            priors.append(l_prior)
                            param_labels.append(f'{kern[i]}_{string}_{j+1}')
                        priors.append(alpha_prior)
                        param_labels.append(f'{kern[i]}_alpha')

        return priors, param_labels
    
    # input warp
    def input_warp(self, X, theta, model):
        """
        Input warping
        
        Parameters:
        X : ndarray
            Input array
        theta : list
            List of input warping parameters
        model : str
            String flag to set GP on mean or variance (noise) model data

        Returns:
        ndarray
            Warped inputs using kumaraswamy cdf.

        """

        # Set parameters
        if model == 'noise':
            iw_params = theta[0:self.iw_noise_idx]
        elif model == 'mean':
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
    def output_warp(self, y, theta, model, revert=False):
        """
        Output warping
        
        Parameters:
        y : ndarray
            Output array
        theta : list
            List of output warping parameters
        model : str
            String flag to set GP on mean or variance (noise) model data
        revert : Logical flag
            Flag to revert back to real space
        Returns:
        ndarray
            Warped outputs using user defined transforms.

        """

        # Set class for noise model parameters
        if model == 'noise':
            ow_params = theta[self.iw_noise_idx:self.ow_noise_idx]
            self.owc_noise = output_warp(warpings=self.ow_noise, params=ow_params)
            y_warped = self.owc_noise.transform(y)
            jac = self.owc_noise.Jacobian(y)
            if revert:
                y_reversed = self.owc_noise.inverse(y)
                return y_reversed
            else:
                return y_warped, jac

        # Set class for mean model parameters
        elif model == 'mean':
            ow_params = theta[self.iw_idx:self.ow_idx]
            self.owc = output_warp(warpings=self.ow_model, params=ow_params)
            y_warped = self.owc.transform(y)
            jac = self.owc.Jacobian(y)
            if revert:
                y_reversed = self.owc.inverse(y)
                return y_reversed
            else:
                return y_warped, jac

    # Get noise kernel
    def set_kernel(self, X_a, X_b, theta, model):
        """
        Set custom kernel
        
        X_a : ndarray
            First input array.
        X_b : ndarray
            Second input array.
        theta : list
            List of hyperparameters for the kernels.
        model : str
            String flag to set GP on mean or variance (noise) model data

        Returns:
        ndarray
            The resulting kernel matrix.

        """

        # Hyper-paramters stored after warp and noise params
        if model == 'noise':
            hypers = theta[self.hypers_noise_idx:]
            kern = self.kern_var
            kern_ops = self.kern_var_ops        
        elif model == 'mean':
            hypers = theta[self.hypers_idx:]
            kern = self.kern
            kern_ops = self.kern_ops

        if self.iw:
            # Warp inputs
            X_a = self.input_warp(X_a, theta, model)
            X_b = self.input_warp(X_b, theta, model)

        # Make kernel
        if self.n_inputs == 1:
            K = kernels.make_kernel(kern, kern_ops, X_a, X_b, hypers)
        elif self.n_inputs > 1:
            K = kernels.make_kernel_nD(kern, kern_ops, X_a, X_b, hypers)
        return K
    

    def monte_carlo_var_array(self, noise_var, theta, num_samples=10000):
        """
        Estimate the variance of transformed variables tilde_y = g(y) for an array of input variances.

        
        noise_var : ndarray
            Noise variance prediction in real space
        theta : list
            List of all hyperparameters.
        num_samples: integer 
            Number of Monte Carlo samples per variance estimation
    
        Returns:
            List of estimated variances [Var(tilde_y1), Var(tilde_y2), ...]
        """
        
        var_tilde_y_array = []
        for var_y in noise_var:
            # Generate y samples from N(mu_y, var_y)
            y_samples = np.random.normal(loc=0, scale=np.sqrt(var_y), size=num_samples)
            y_samples = np.where(y_samples < 0, 1e-8, y_samples)  # Set negatives to near zero (not at zero as log 0 gives issue)
            # Apply the transformation g(y)
            tilde_y_samples, _ = self.output_warp(y_samples, theta, model='mean') 
            # Compute and store the variance of transformed samples
            if np.isnan(np.var(tilde_y_samples)):
                print(y_samples)
            #     print(np.var(tilde_y_samples))
            #     print('-----------------------------')

        
            var_tilde_y_array.append(np.var(tilde_y_samples))
        return np.array(var_tilde_y_array)
    
    # Get noise model weights and kernel
    def update_gp(self, theta, model):
        """
        Update Gaussian Process
        
        theta : list
            List of all hyperparameters.
        model : str
            String flag to set GP on mean or variance (noise) model data

        """

        if model == 'noise':
            # Noise
            self.sigma_n = theta[self.ow_noise_idx:self.hypers_noise_idx]
            if self.ow_noise != None:
                # Output warp
                self.y_warp, self.jac = self.output_warp(self.y_var_train, theta, model)
            else:
                self.y_warp = self.y_var_train
                self.jac = 1 # Log(jac) = 0 thus recovers old log marginal likelihood
            # Input 
            X = self.X_var_train

        elif model == 'mean':
            # Noise
            self.sigma_n = theta[self.ow_idx:self.hypers_idx]
            if self.ow_model != None:
                # Output warp
                self.y_warp, self.jac = self.output_warp(self.y_train, theta, model)
            else:
                self.y_warp = self.y_train
                self.jac = 1 # Log(jac) = 0 thus recovers old log marginal likelihood
            # Input 
            X = self.X_train


        # Kernel
        self.K = self.set_kernel(X, X, theta, model)
        self.K += np.eye(len(X)) * self.sigma_n
        # NOT QUITE WORKING AS THERE IS COMPLICATIONS WITH PREDICTIONS WITH WARPINGS, USE BASIC ADDITIONS OF NOISE FOR NOW
        # if model == 'mean':
        #     var = self.posterior_predict(X, model='noise', scale=False, get_var=False)
        #     var_warp = self.monte_carlo_var_array(var, theta, num_samples=100)
        #     # print(var_warp)
        #     self.K += np.diag(var_warp)

        try:
            # Weights
            self.L = np.linalg.cholesky(self.K)
        except np.linalg.LinAlgError:
            sys.exit('Defined kernel is NOT positive defnite, please make another kernel')
        self.weights = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_warp))
    
    # Predictive posterior
    def posterior_predict(self, X_star, model, scale=False, get_var=False):
        
        """
        Inference at new input locations.

        X_star : ndarray
            New inputs to infer at.
        model : str
            String flag to set GP on mean or variance (noise) model data
        scale : logical flag
            Flag to re-scale output to real space (not warped space).
        get_var : logical flag
            Flag to output variance.
        
        Returns:
        ndarray
            The posterior predictive mean and var (if get_var=True).
        """

        if X_star.ndim == 1:
            X_star = X_star[:, None]  # Reshape (n,) to (n,1)

        if X_star.shape[-1] != self.n_inputs:
            sys.exit('(ERROR) : Number of inputs at new locations does match that of the training data')

        # Re-scale
        if scale:
            X_sc = np.zeros_like(X_star)
            for i in range(self.n_inputs):
                sc = zero_one_scale(eps=0.01, x_max=np.max(self.X[:,i]), x_min=np.min(self.X[:,i]))
                X_sc[:,i] = sc.transform(X_star[:,i])
            X_star = X_sc
        
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

        """
        Function to evaluate inference

        X_star : ndarray
            New inputs to infer at.
        model : str
            String flag to set GP on mean or variance (noise) model data
        
        Returns:
        ndarray
            The posterior predictive mean and variance
        """

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
        """
        log likelihood optimiser function

        X_star : ndarray
            New inputs to infer at.
        
        Returns:
        function
            Log likelihood calculator
            
        """
        # Extract log likelihood to minimise/maximise
        def get_log_likelihood(theta):
            """
            log likelihood function

            theta : list
            List of all hyperparameters.
        
            Returns:
            ndarray
                Negative Log marginal likelihood
            
        """
            self.update_gp(theta, self.model)
            sign, logdet = np.linalg.slogdet(self.K)
            n = len(self.K.diagonal())
            log_L = -0.5*np.dot(self.y_warp.T, self.weights) - 0.5*logdet - 0.5*n*np.log(2*np.pi) + np.sum(np.log(self.jac))
            return -1.0*log_L
        return get_log_likelihood

    # Optimisation routine using scipy minimize optimize or differential evoloution
    def optimise_gp(self, model, solver='opt', n_restarts=10, method='L-BFGS-B', max_iter=5000, strategy='best1bin', tol=1e-6, save=True, fname='mean_model_nD.pkl'):
        """
        Optimising hyperparamter routine

        model : str
            String flag to set GP on mean or variance (noise) model data
        solver : str
            String to set either minimize ('opt') or differential evoloution ('diff_evo') method.
        n_restats : int
            Number of minimize restarts to avoid local minima (for solver='opt')
        method : str
            Optimisation method for minimize to use (for solver='opt')
        max_iter : int
            Maximum number of iterations (for solver='diff_evo')
        strategy : str
            Differential evoloution method (for solver='diff_evo')
        tol : float
            Accepted tolerannce for differential evoloution method (for solver='diff_evo')
        save : logical flag
            Flag to make a save file with all required GP data to store
        fname : str
            Filename to store GP setup data to.
            
        """

        # Set model 
        self.model = model

        # Priors
        priors, param_labels = self.set_priors(model)
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
            
                print(f'restart {n+1} = {res.fun}')
            
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
            self.theta_noise_labels = param_labels
            if save:
                data = {"theta": self.theta_noise,
                        "theta_labels": self.theta_noise_labels,
                        "X": self.X,
                        "y": self.y_var,
                        "y_test": self.y_var_test,
                        "y_train": self.y_var_train,
                        "X_test": self.X_var_test,
                        "X_train": self.X_var_train,
                        "kern": self.kern_var,
                        "kern_ops": self.kern_var_ops,
                        "iw" : self.iw,
                        "ow_model": self.ow_noise,
                        "iw_idx": self.iw_noise_idx,
                        "ow_idx": self.ow_noise_idx,
                        "hypers_idx" : self.hypers_noise_idx}
                # Save to a pickle file
                with open("noise_model_2D.pkl", "wb") as file:
                    pickle.dump(data, file)
                print('Saved parameters to noise_model_2D.pkl')
        
        elif model == 'mean':
            self.theta = theta
            self.theta_labels = param_labels
            if save:
                data = {"theta": self.theta,
                        "theta_labels": self.theta_labels,
                        "X": self.X,
                        "y": self.y,
                        "y_test": self.y_test,
                        "y_train": self.y_train,
                        "X_test": self.X_test,
                        "X_train": self.X_train,
                        "kern": self.kern,
                        "kern_ops": self.kern_ops,
                        "iw" : self.iw,
                        "ow_model": self.ow_model,
                        "iw_idx": self.iw_idx,
                        "ow_idx": self.ow_idx,
                        "hypers_idx" : self.hypers_idx}
                # Save to a pickle file
                with open("mean_model_2D.pkl", "wb") as file:
                    pickle.dump(data, file)
                print('Saved parameters to mean_model_2D.pkl')

    #############################################################
    # Set parmaters manually
    #############################################################
    
    # Read in paramters dictionary with theta and indicies 
    def read_gp_model(self, file, model):

        """
        Read in GP model from previous save file.
        
        file : str
            Path to file storing GP setup data.
        model : str
            String flag to set GP on mean or variance (noise) model data
            
        """
        
        with open(file, "rb") as file:
            data = pickle.load(file)
        
        if model == 'noise':
            self.theta_noise = data['theta']
            self.theta_noise_labels = data["theta_labels"]
            self.X = data["X"]
            self.y_var = data["y"]
            self.y_var_test = data["y_test"]
            self.y_var_train = data["y_train"]
            self.X_var_test = data["X_test"]
            self.X_var_train = data["X_train"]
            self.kern_var = data["kern"]
            self.kern_var_ops = data["kern_ops"]
            self.iw = data["iw"]
            self.ow_noise = data["ow_model"]
            self.iw_noise_idx = data['iw_idx']
            self.ow_noise_idx = data['ow_idx']
            self.hypers_noise_idx = data['hypers_idx']
        elif model == 'mean':
            self.theta = data['theta']
            self.theta_labels = data["theta_labels"]
            self.X = data["X"]
            self.y = data["y"]
            self.y_test = data["y_test"]
            self.y_train = data["y_train"]
            self.X_test = data["X_test"]
            self.X_train = data["X_train"]
            self.kern = data["kern"]
            self.kern_ops = data["kern_ops"]
            self.iw = data["iw"]
            self.ow_model = data["ow_model"]
            self.iw_idx = data['iw_idx']
            self.ow_idx = data['ow_idx']
            self.hypers_idx = data['hypers_idx']
        
        self.n_inputs = self.X.shape[-1]
        self.n_iw_params = int(2*self.n_inputs)
        
        # scale inputs between 0 and 1
        self.X_sc = np.zeros_like(self.X)
        for i in range(self.n_inputs):
            sc = zero_one_scale(eps=0.01, x_max=np.max(self.X[:,i]), x_min=np.min(self.X[:,i]))
            self.X_sc[:,i] = sc.transform(self.X[:,i])
            
        # Update GP variables
        self.update_gp(data['theta'], model)
  
    
    #############################################################
    # Gauss - Hermite quadrature
    #############################################################
    def gauss_hermite_quad(self, mu, var, model, deg=8):

        """
        Gauss-Hermite quadrature to restore output to real space predictions.

        mu : ndarray
            Posterior predicted mean in warped space
        var : ndarray
            Posterior predicted variance in warped space
        model : str
            String flag to set GP on mean or variance (noise) model data
        deg : int
            Degree of Gauss-Hermite quadrature to use.

        
        Returns:
        ndarray
            returns the real space mean and variance
        """

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
    def test_train_plots(self, model, fname=f'test_train.png'):

        """
        Plot test-train plots for GP prediction

        model : str
            String flag to set GP on mean or variance (noise) model data
        fname : str
            Filename to store GP setup data to.
        """
        
        fig = plt.figure()

        # Set correct model params
        if model == 'noise':
            y = self.y_var
            X_train = self.X_var_train
            X_test = self.X_var_test
            y_train = self.y_var_train
            y_test = self.y_var_test

            # train and test predictions
            y_train_predict, var_train = self.posterior_predict(X_train, model, scale=False, get_var=True) 
            y_test_predict, var_test = self.posterior_predict(X_test, model, scale=False, get_var=True) 
        
        elif model == 'mean':
            y = self.y
            X_train = self.X_train
            X_test = self.X_test
            y_train = self.y_train
            y_test = self.y_test

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
        ax3.set_xlabel(r'Output Values')

        ax1.scatter(y_train, y_train_predict, label=f'Train (RSME = {np.round(rmse_train, 3)})', color = 'blue')
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k:', label = 'Target')
        ax1.set_xlabel(r'True Value')
        ax1.set_ylabel(r'Predicted Value')
        ax2.plot(abs(y_train_predict - y_train), S_ptrain, 'o', label='Train', color = 'blue')
        ax2.plot([0, S_ptrain.max()], [0, S_ptrain.max()], 'k:', label = 'Target')
        ax1.scatter(y_test, y_test_predict, label=f'Test (RSME = {np.round(rmse_test, 3)})', color = 'orange')
        ax1.legend()
        ax2.plot(abs(y_test_predict - y_test), S_ptest, 'o', label='Test', color='orange')
        ax2.plot([0, S_ptest.max()], [0, S_ptest.max()], 'k:', label = 'Target')
        ax2.legend()
        ax2.set_xlabel(r'True Error')
        ax2.set_ylabel(r'Predicted Error')
        ax1.set_ylim(y.min(), y.max())
        # Make plots spaced out
        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.5, 
                            hspace=0.4)

        plt.savefig(fname, bbox_inches='tight')

    
