import numpy as np
import scipy.spatial
from scipy.special import kv, gamma
import scipy
import sys


######################################################
# Basic kernels
######################################################

def matern_kernel(X_a, X_b, hyper_param=[1, 1], nu=1.5, ARD=False):
    """
    Computes the Matern Kernel for inputs X_a and X_b.
    
    Parameters:
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    hyper_param : list, optional
        List of hyperparameters [variance, length-scale]. Default is [1, 1].
    nu : float, optional
        Smoothness parameter. Default is 1.5.
    ARD : logical flag
        Use ARD detection (seperate length scales per dimesnion need hyper_param have len(N_iputs) + 1)

    Returns:
    ndarray
        The computed Matern kernel matrix.
    """
    
    # Euclidean distance
    dist = scipy.spatial.distance.cdist(X_a, X_b, metric='euclidean')
    
    # Exponential kernel
    if nu == 0.5:
        if ARD == True:
            K = hyper_param[0]
            for i in range(X_a.shape[-1]):
                # Inputs 
                X_1 = X_a[:,i][:,None]
                X_2 = X_b[:,i][:,None]
                # Euclidean distance
                dist = scipy.spatial.distance.cdist(X_1, X_2, metric='euclidean')
                
                K *= np.exp(-dist / hyper_param[i+1])
            return K                
        else:
            return hyper_param[0] * np.exp(-dist / hyper_param[1])
    
    # Matern 3_2
    elif nu == 1.5:
        if ARD == True:
            K = hyper_param[0]
            for i in range(X_a.shape[-1]):
                # Inputs 
                X_1 = X_a[:,i][:,None]
                X_2 = X_b[:,i][:,None]
                # Euclidean distance
                dist = scipy.spatial.distance.cdist(X_1, X_2, metric='euclidean')
                
                factor = np.sqrt(3) * dist / hyper_param[i+1]
                K *= ((1 + factor) * np.exp(-factor))
            return K                
        else:
            factor = np.sqrt(3) * dist / hyper_param[1]
            return hyper_param[0] * (1 + factor) * np.exp(-factor)
    
    # Matern 5_2
    elif nu == 2.5:
        if ARD == True:
            K = hyper_param[0]
            for i in range(X_a.shape[-1]):
                # Inputs 
                X_1 = X_a[:,i][:,None]
                X_2 = X_b[:,i][:,None]
                # Euclidean distance
                dist = scipy.spatial.distance.cdist(X_1, X_2, metric='euclidean')
                
                factor = np.sqrt(5) * dist / hyper_param[i+1]
                K *= ((1 + factor + (5 * dist ** 2) / (3 * hyper_param[i+1] ** 2)) * np.exp(-factor))
            return K       
        else:
            factor = np.sqrt(5) * dist / hyper_param[1]
            return hyper_param[0] * (1 + factor + (5 * dist ** 2) / (3 * hyper_param[1] ** 2)) * np.exp(-factor)
    
    # General
    else:
        if ARD == True:
            K = hyper_param[0]
            for i in range(X_a.shape[-1]):
                # Inputs 
                X_1 = X_a[:,i][:,None]
                X_2 = X_b[:,i][:,None]
                # Euclidean distance
                dist = scipy.spatial.distance.cdist(X_1, X_2, metric='euclidean')
                
                factor = (np.sqrt(2 * nu) * dist) / hyper_param[i+1]
                K *= ((2 ** (1 - nu) / gamma(nu)) * (factor ** nu) * kv(nu, factor) if dist > 0 else 1.0)
            return K
        else:
            factor = (np.sqrt(2 * nu) * dist) / hyper_param[1]
            return hyper_param[0] * (2 ** (1 - nu) / gamma(nu)) * (factor ** nu) * kv(nu, factor) if dist > 0 else 1.0


def rbf_kernel(X_a, X_b, hyper_param=[1, 1], ARD=False):
    """
    Computes the radial basis function kernel for inputs X_a and X_b.
    
    Parameters:
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    hyper_param : list, optional
        List of hyperparameters [variance, length-scale]. Default is [1, 1].
    ARD : logical flag
        Use ARD detection (seperate length scales per dimesnion need hyper_param have len(N_iputs) + 1)

    Returns:
    ndarray
        The computed RBF kernel matrix.
    """
    
    # Square Euclidean distance
    sqdist = scipy.spatial.distance.cdist(X_a, X_b, metric='sqeuclidean')

    if ARD == True:
            K = hyper_param[0]
            for i in range(X_a.shape[-1]):
                # Inputs 
                X_1 = X_a[:,i][:,None]
                X_2 = X_b[:,i][:,None]
                # Euclidean distance
                sqdist = scipy.spatial.distance.cdist(X_1, X_2, metric='sqeuclidean')
                
                K *= np.exp(-0.5 * sqdist / hyper_param[i+1]**2) 
            return K
    else:
        return hyper_param[0] * np.exp(-0.5 * sqdist / hyper_param[1]**2)


def rat_quad_kernel(X_a, X_b, hyper_param=[1, 1, 1], ARD=False):
    """
    Computes the rational quadratic kernel for inputs X_a and X_b.
    
    Parameters:
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    hyper_param : list, optional
        List of hyperparameters [variance, length-scale, alpha]. Default is [1, 1, 1].
    ARD : logical flag
        Use ARD detection (seperate length scales per dimesnion need hyper_param have len(N_iputs) + 2)

    Returns:
    ndarray
        The computed rational quadratic kernel matrix.
    """
    
    sqdist = scipy.spatial.distance.cdist(X_a, X_b, metric='sqeuclidean')
    if ARD == True:
            K = hyper_param[0]
            for i in range(X_a.shape[-1]):
                # Inputs 
                X_1 = X_a[:,i][:,None]
                X_2 = X_b[:,i][:,None]
                # Euclidean distance
                sqdist = scipy.spatial.distance.cdist(X_1, X_2, metric='sqeuclidean')
                
                K *= (1.0 + 0.5*sqdist/hyper_param[i+1]**2/hyper_param[-1])**(-1.0*hyper_param[-1])
            return K
    else:
        return hyper_param[0] * (1.0 + 0.5*sqdist/hyper_param[1]**2/hyper_param[2])**(-1.0*hyper_param[2])


######################################################
# Non-separable kernels
######################################################

def mahalanobis_distance(X_a, X_b, Lambda_inv):
    """Compute Mahalanobis distance matrix."""
    # Use cdist to compute pairwise Mahalanobis distance
    distances = scipy.spatial.distance.cdist(X_a, X_b, metric='mahalanobis', VI=Lambda_inv)
    return distances

def matern_kernel_NS(X_a, X_b, var, Lambda_inv, nu=1.5):
    """
    Computes the non-separable Matern Kernel for inputs X_a and X_b.
    
    Parameters:
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    var : float
        Kernel variance.
    Lambda_inv : ndarray
        Inverse of covariance matrix.
    nu : float, optional
        Smoothness parameter. Default is 1.5.

    Returns:
    ndarray
        The computed non-separable Matern kernel matrix.
    """
    
    # Mahalanobis distance
    # dist = scipy.spatial.distance.mahalanobis(X_a, X_b, Lambda_inv)
    dist = mahalanobis_distance(X_a, X_b, Lambda_inv)
    
    # Exponential kernel
    if nu == 0.5:
        return var * np.exp(-dist)
    # Matern 3_2
    elif nu == 1.5:
        factor = np.sqrt(3) * dist
        return var * (1 + factor) * np.exp(-factor)
    # Matern 5_2
    elif nu == 2.5:
        factor = np.sqrt(5) * dist
        return var * (1 + factor + (5/3) * dist**2) * np.exp(-factor)
    # General
    else:
        factor = np.sqrt(2 * nu) * dist
        return var * (2 ** (1 - nu) / gamma(nu)) * (factor ** nu) * kv(nu, factor) if dist > 0 else 1.0


def rbf_kernel_NS(X_a, X_b, var, Lambda_inv):
    """
    Computes the non-separable radial basis function Kernel for inputs X_a and X_b.
    
    Parameters:
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    var : float
        Kernel variance.
    Lambda_inv : ndarray
        Inverse of covariance matrix.

    Returns:
    ndarray
        The computed non-separable RBF kernel matrix.
    """
    
    # Square Mahalanobis distance
    # sqdist = scipy.spatial.distance.mahalanobis(X_a, X_b, Lambda_inv)**2
    sqdist = mahalanobis_distance(X_a, X_b, Lambda_inv)**2
    return var * np.exp(-0.5 * sqdist)


def rat_quad_kernel_NS(X_a, X_b, var, Lambda_inv, alpha):
    """
    Computes the non-separable rational quadratic Kernel for inputs X_a and X_b.
    
    Parameters:
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    var : float
        Kernel variance.
    Lambda_inv : ndarray
        Inverse of covariance matrix.
    alpha : float
        Parameter controlling the relative scale of the distance.

    Returns:
    ndarray
        The computed non-separable rational quadratic kernel matrix.
    """
    
    # Mahalanobis distance
    # sqdist = scipy.spatial.distance.mahalanobis(X_a, X_b, Lambda_inv)**2
    sqdist = mahalanobis_distance(X_a, X_b, Lambda_inv)**2
    return var * (1 + (sqdist / (2 * alpha)))**(-alpha)


######################################################
# Make custom kernels
######################################################

def make_kernel(kern_labels, kern_ops, X_a, X_b, hypers):
    """
    Make kernel from either one or a combination of kernels for the case of
    1D inputs.
    
    Parameters:
    kern_labels : list of str
        List of kernel labels ('RBF', 'EXP', 'MATERN_3_2', 'MATERN_5_2', 'RAT_QUAD').
    kern_ops : str
        String defining overll kernel e.g "k_1 * (k_2 + k_3) + k_4".
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    hypers : list
        List of hyperparameters for the kernels.

    Returns:
    ndarray
        The resulting combined kernel matrix.
    """
    
    # set index for hyperparameters
    idx = 0

    # Store kernels
    kernels = {}
    
    # Set kernel
    for i in range(len(kern_labels)):
        if kern_labels[i] == 'RBF':
            K_i = rbf_kernel(X_a, X_b, hyper_param=hypers[idx:idx+2])
            idx += 2
        elif kern_labels[i] == 'EXP':
            K_i = matern_kernel(X_a, X_b, hyper_param=hypers[idx:idx+2], nu=0.5) 
            idx += 2
        elif kern_labels[i] == 'MATERN_3_2':
            K_i = matern_kernel(X_a, X_b, hyper_param=hypers[idx:idx+2], nu=1.5)
            idx += 2 
        elif kern_labels[i] == 'MATERN_5_2':
            K_i = matern_kernel(X_a, X_b, hyper_param=hypers[idx:idx+2], nu=2.5)
            idx += 2
        elif kern_labels[i] == 'RAT_QUAD':
            K_i = rat_quad_kernel(X_a, X_b, hyper_param=hypers[idx:idx+3])
            idx += 3
        
        # Add kernel to dictionary
        kernels[f"k_{i+1}"] = K_i
            
    # Now perform operations to define true kernel
    kernel = eval_kernel_expr(kern_ops, kernels)

    return kernel

def make_kernel_nD(kern_labels, kern_ops, X_a, X_b, hypers):
    """
    Make kernel from either one or a combination of kernels for the case of nD inputs.
    
    Parameters:
    kern_labels : list of str
        List of kernel labels ('RBF', 'EXP', 'MATERN_3_2', 'MATERN_5_2', 'RAT_QUAD').
    kern_ops : str
        String defining overll kernel e.g "k_1 * (k_2 + k_3) + k_4".
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    hypers : list
        List of hyperparameters for the kernels.

    Returns:
    ndarray
        The resulting combined kernel matrix.
    """
    
    
    idx = 0
    check_array = []
    kernels = {}

    for i in range(len(kern_labels)):
        val = extract_numbers_after_kernel(kern_labels[i])
        if val == None:
            check_array.append(val)
        elif isinstance(val, list) and all(isinstance(x, int) for x in val):
            check_array.extend(val)
        else:
            sys.exit('(ERROR) Issue with naming. Please use the followinf form e.g RBF, RBF_ISO_[1,2..], RBF_NS, RBF_NS_[1, 2..], RBF_ARD, RBF_ARD_[1,2..]')
    check = check_dim_nums(check_array)

    check_dims = extract_numbers(kern_labels)
    if all(isinstance(x, int) and 0 <= x <= X_a.shape[-1] for x in check_dims) == False:
        sys.exit('(ERROR) All specified dimensions should be an integer between 1 and N_inputs')
    
    ###########################################################################
    # Either isotropic or non-seperable (No dims so across all dimensions)
    ###########################################################################

    if check == "all_none":
        non_sep = check_NS(kern_labels)
        ard = check_ARD(kern_labels)
        isotropic = check_ISO(kern_labels)

        if isotropic:
            sys.exit('(ERROR) To define isotopic along all dimensions remove _IS flag in kernel labels. The _IS flag is only for using certain dimesnions, given by the numbers after the flag.')

        # All kernels are isotropic (correctly labelled) across all dims so can handle like 1D
        elif non_sep == False and ard == False:
            kernel = make_kernel(kern_labels, kern_ops, X_a, X_b, hypers)
    
        # Can be all or mixture of non-seperable, ARD and isotropic (across all or subset)
        else:
            for i in range(len(kern_labels)):
                # Check kernel type
                k_type = kernel_type(kern_labels[i])
                label = get_kernel_label(kern_labels[i])
                # Isotropic or ARD
                if k_type == None or k_type == '_ARD':
                    
                    # Isotropic across all dimensions
                    if k_type == None:
                        flag = False
                        param_move = 2 # Variance and length-scale
                    else:
                        flag = True
                        param_move = X_a.shape[-1] + 1 # Variance and length-scales
                    
                    if label == 'RBF':
                        K_i = rbf_kernel(X_a, X_b, hyper_param=hypers[idx:idx+param_move], ARD=flag)
                        idx += param_move
                    elif label == 'EXP':
                        K_i = matern_kernel(X_a, X_b, hyper_param=hypers[idx:idx+param_move], nu=0.5, ARD=flag) 
                        idx += param_move
                    elif label == 'MATERN_3_2':
                        K_i = matern_kernel(X_a, X_b, hyper_param=hypers[idx:idx+param_move], nu=1.5, ARD=flag)
                        idx += param_move
                    elif label == 'MATERN_5_2':
                        K_i = matern_kernel(X_a, X_b, hyper_param=hypers[idx:idx+param_move], nu=2.5, ARD=flag)
                        idx += param_move
                    elif label == 'RAT_QUAD':
                        K_i = rat_quad_kernel(X_a, X_b, hyper_param=hypers[idx:idx+param_move+1], ARD=flag)
                        idx += param_move + 1 # Will be +1 for RAT_QUAD beacuse of alpha
                    else:
                        sys.exit(f'(ERROR) Kernel label is not recognised {label}')
        
                # Non-seperable case
                elif k_type == '_NS':
                    
                    # How many hyper-parameters to move afterwards for next part
                    param_move = int(0.5*X_a.shape[-1]*(X_a.shape[-1]+1))
                    # Fill lower triangular matrix
                    L = np.zeros((X_a.shape[-1], X_a.shape[-1]))
                    L[np.tril_indices(X_a.shape[-1])] = hypers[idx+1:param_move+1]
                    # Compute inverse covariance
                    Lambda = np.dot(L, L.T)
                    Lambda_inv = np.linalg.inv(Lambda)
     
                    if label == 'RBF':
                        K_i = rbf_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv)
                        idx += param_move + 1
                    elif label == 'EXP':
                        K_i = matern_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv, nu=0.5)
                        idx += param_move + 1
                    elif label == 'MATERN_3_2':
                        K_i = matern_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv, nu=1.5)
                        idx += param_move + 1
                    elif label == 'MATERN_5_2':
                        K_i = matern_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv, nu=2.5)
                        idx += param_move + 1
                    elif label == 'RAT_QUAD':
                        K_i = rat_quad_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv, hypers[idx+param_move+1])
                        idx += param_move + 2
                    else:
                        sys.exit(f'(ERROR) Kernel label is not recognised {label}')   

                # Add kernel to dictionary
                kernels[f"k_{i+1}"] = K_i
                

        
    
    #############################################################################################
    # Either all Mixture of isotropic, non-seperable, seperable (either dimension specific or not)
    #############################################################################################

    elif check == "integers_and_none" or check == "all_integers":
        if check == "all_integers" and set(range(1, X_a.shape[-1] + 1)).issubset(set(check_dims)) == False:
            sys.exit('(ERROR) All input dimensions should be used in kernel description')
        else:
            for i in range(len(kern_labels)):   
                # Check kernel type
                k_type = kernel_type(kern_labels[i])
                label = get_kernel_label(kern_labels[i])
                dims = extract_numbers_after_kernel(kern_labels[i])
                # Isotropic over-all dimesnions or only a select few
                if k_type in {"_ISO", "_ARD", None}:

                    if k_type == None:
                        flag = False
                        param_move = 2
                        # Full isotropic
                        if dims == None:
                            X_1 = X_a
                            X_2 = X_a
                        # Seperable with given dimensions
                        elif len(dims) == 1:
                            X_1 = X_a[:,dims[0]-1][:,None]
                            X_2 = X_b[:,dims[0]-1][:,None]
                        elif len(dims) == 2:
                            X_1 = X_a[:,dims[0]-1][:,None]
                            X_2 = X_b[:,dims[1]-1][:,None]
                        elif len(dims) > 2:
                            sys.exit('(ERROR) : If using seperable kernel to compute across dimesnions then only 2 numbers should be given after kernel label')

                    elif k_type == "_ISO":
                        flag = False
                        param_move = 2
                        if len(dims) == 1:
                            sys.exit('(ERROR) If using isotropic key (_ISO) then please ensure no dimesnions are given, or more than 1 are e.g _ISO_[1, 2, ..],  where len(arr) > 1.')
                        elif len(dims) == len(set(dims)) == False:
                            sys.exit('(ERROR) Repated dimension in isotropic kernel description, believes remove repated value.')
                        elif len(dims) > X_a.shape[-1]:
                            sys.exit('(ERROR) Cannot specify more input dimensions than there are.')
                        elif dims == None:
                            X_1 = X_a
                            X_2 = X_b 
                        else:
                            use_dims = []
                            for d in dims:
                                use_dims.append(d-1)
                            X_1 = X_a[:, use_dims]
                            X_2 = X_b[:, use_dims]
    
                    elif k_type == "_ARD":
                        flag = True
                        if len(dims) == 1:
                            sys.exit('(ERROR) If using ARD key (_ARD) then please ensure no dimesnions are given, or more than 1 are e.g _ARD_[1, 2, ..],  where len(arr) > 1.')
                        elif len(dims) == len(set(dims)) == False:
                            sys.exit('(ERROR) Repated dimension in isotropic kernel description, believes remove repated value.')
                        elif len(dims) > X_a.shape[-1]:
                            sys.exit('(ERROR) Cannot specify more input dimensions than there are.')
                        elif dims == None:
                            X_1 = X_a
                            X_2 = X_b 
                            param_move = X_1.shape[-1] + 1 
                        else:
                            use_dims = []
                            for d in dims:
                                use_dims.append(d-1)
                            X_1 = X_a[:, use_dims]
                            X_2 = X_b[:, use_dims]
                            param_move = X_1.shape[-1] + 1 
                    
                    if label == 'RBF':
                        K_i = rbf_kernel(X_1, X_2, hyper_param=hypers[idx:idx+param_move], ARD=flag)
                        idx += param_move
                    elif label == 'EXP':
                        K_i = matern_kernel(X_1, X_2, hyper_param=hypers[idx:idx+param_move], nu=0.5, ARD=flag) 
                        idx += param_move
                    elif label == 'MATERN_3_2':
                        K_i = matern_kernel(X_1, X_2, hyper_param=hypers[idx:idx+param_move], nu=1.5, ARD=flag)
                        idx += param_move
                    elif label == 'MATERN_5_2':
                        K_i = matern_kernel(X_1, X_2, hyper_param=hypers[idx:idx+param_move], nu=2.5, ARD=flag)
                        idx += param_move
                    elif label == 'RAT_QUAD':
                        K_i = rat_quad_kernel(X_1, X_2, hyper_param=hypers[idx:idx+param_move+1], ARD=flag)
                        idx += param_move + 1 # Will be +1 for RAT_QUAD beacuse of alpha
                    else:
                        sys.exit(f'(ERROR) Kernel label is not recognised {label}')
                
                elif k_type == '_NS':

                    if dims == None:
                        X_1 = X_a
                        X_2 = X_b
                    else:
                        use_dims = []
                        for d in dims:
                            use_dims.append(d-1)
                        X_1 = X_a[:, use_dims]
                        X_2 = X_b[:, use_dims]
                    
                    # How many hyper-parameters to move afterwards for next part
                    param_move = int(0.5*X_1.shape[-1]*(X_1.shape[-1]+1))

                    # Fill lower triangular matrix
                    L = np.zeros((X_1.shape[-1], X_1.shape[-1]))
                    L[np.tril_indices(X_1.shape[-1])] = hypers[idx+1:param_move+1]
                    # Compute inverse covariance
                    Lambda = np.dot(L, L.T)
                    Lambda_inv = np.linalg.inv(Lambda)
     
                    if label == 'RBF':
                        K_i = rbf_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv)
                        idx += param_move + 1
                    elif label == 'EXP':
                        K_i = matern_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv, nu=0.5)
                        idx += param_move + 1
                    elif label == 'MATERN_3_2':
                        K_i = matern_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv, nu=1.5)
                        idx += param_move + 1
                    elif label == 'MATERN_5_2':
                        K_i = matern_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv, nu=2.5)
                        idx += param_move + 1
                    elif label == 'RAT_QUAD':
                        K_i = rat_quad_kernel_NS(X_a, X_b, hypers[idx], Lambda_inv, hypers[idx+param_move+1])
                        idx += param_move + 2
                    else:
                        sys.exit(f'(ERROR) Kernel label is not recognised {label}')

                # Add kernel to dictionary
                kernels[f"k_{i+1}"] = K_i
    else:
        sys.exit('(ERROR) Naming of kernels is incorrect')

    # Now perform operations to define true kernel
    kernel = eval_kernel_expr(kern_ops, kernels)

    return kernel
                        

################################################################
# Check functions
################################################################

def check_NS(kern_arr):
    return any("_NS" in s for s in kern_arr)

def check_ARD(kern_arr):
    return any("_ARD" in s for s in kern_arr)

def check_ISO(kern_arr):
    return any("_ISO" in s for s in kern_arr)

def kernel_type(kernel_str):
    keywords = ["_NS", "_ARD", "_ISO"]
    for keyword in keywords:
        if keyword in kernel_str:
            return keyword 
    return None

def get_kernel_label(kernel_str):
    kernels=('RBF', 'EXP', 'MATERN_3_2', 'MATERN_5_2', 'RAT_QUAD')
    for kernel in kernels:
        if kernel_str.startswith(kernel):
            return kernel
    return None  # Return None if no match is found

def extract_numbers_after_kernel(s):
    if "[" in s and "]" in s:
        # Extract substring inside brackets
        s = s[s.index("[") + 1 : -1]
        # Split and convert to integers
        return [int(num) for num in s.split(",") if num.strip().isdigit()]
    return None

def check_dim_nums(lst):
    if all(item is None for item in lst):
        return "all_none"
    elif all(isinstance(item, int) for item in lst):
        return "all_integers"
    elif all(isinstance(item, int) or item is None for item in lst):
        return "integers_and_none"
    else:
        return "mixed_types"


def extract_numbers(arr):
    exclusions = ['EXP', 'MATERN_3_2', 'MATERN_5_2', 'RBF', 'RAT_QUAD']
    numbers = []

    for item in arr:
        # Check if any exclusion phrase is part of the item
        for exclusion in exclusions:
            if exclusion in item:
                # If exclusion phrase found, remove any numbers attached to it
                item = item.replace(exclusion, '')  # Remove the exclusion phrase entirely
        
        # Now, extract numbers from the remaining part of the item
        num = ''
        for char in item:
            if char.isdigit():
                num += char  # Build the number as a string
            elif num:
                numbers.append(int(num))  # Convert and store when non-digit appears
                num = ''
        if num:  # Add the last number if it exists
            numbers.append(int(num))

    return numbers

#############################################################
# Kernel calculator
#############################################################

def tokenize(expr):
    tokens = []        # Final list of tokens (e.g. ['k_1', '*', '(' ...])
    i = 0              # Index to walk through the string

    # Loop until we have processed every character
    while i < len(expr):

        # Ignore whitespace entirely
        if expr[i].isspace():
            i += 1

        # If the character is an operator or parenthesis,
        # it is already a complete token
        elif expr[i] in "+*()":
            tokens.append(expr[i])
            i += 1

        # If the character is part of a kernel name
        # (letters, numbers, underscores)
        elif expr[i].isalnum() or expr[i] == "_":
            start = i

            # Keep reading characters until the name ends
            while i < len(expr) and (expr[i].isalnum() or expr[i] == "_"):
                i += 1

            # Extract the full kernel name as one token
            tokens.append(expr[start:i])

        # Any other character is invalid
        else:
            raise ValueError(f"Invalid character: {expr[i]}")

    return tokens

def to_postfix(tokens):
    output = []        # Final postfix expression
    op_stack = []      # Stack to temporarily store operators

    # Define operator precedence
    precedence = {
        "+": 1,
        "*": 2,
    }

    # Process each token in order
    for token in tokens:

        # If the token is a kernel name (operand),
        # send it directly to the output
        if token not in "+*()":
            output.append(token)

        # If the token is an operator
        elif token in "+*":

            # While there is an operator on the stack
            # with greater or equal precedence,
            # pop it to the output
            while (
                op_stack
                and op_stack[-1] in "+*"
                and precedence[op_stack[-1]] >= precedence[token]
            ):
                output.append(op_stack.pop())

            # Push the current operator onto the stack
            op_stack.append(token)

        # Left parenthesis: acts as a barrier
        elif token == "(":
            op_stack.append(token)

        # Right parenthesis: pop until matching "("
        elif token == ")":
            while op_stack and op_stack[-1] != "(":
                output.append(op_stack.pop())

            # Remove the "(" itself
            op_stack.pop()

    # After processing all tokens,
    # pop any remaining operators to output
    while op_stack:
        output.append(op_stack.pop())

    return output


def eval_postfix(postfix, kernels):
    stack = []   # Stack to hold kernel objects

    # Process each token in postfix order
    for token in postfix:

        # If the token is an operator
        if token in "+*":

            # Pop the last two kernels
            b = stack.pop()
            a = stack.pop()

            # Apply the correct operation
            if token == "+":
                stack.append(a + b)
            elif token == "*":
                stack.append(a * b)

        # Otherwise, the token is a kernel name
        else:
            # Push the corresponding kernel object
            stack.append(kernels[token])

    # The final result is the only item left on the stack
    return stack[0]


def eval_kernel_expr(expr, kernels):
    # Step 1: Convert string into tokens
    tokens = tokenize(expr)
    # Step 2: Convert infix tokens to postfix
    postfix = to_postfix(tokens)
    # Step 3: Evaluate postfix expression
    return eval_postfix(postfix, kernels)