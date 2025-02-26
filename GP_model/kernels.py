import numpy as np
import scipy.spatial
from scipy.special import kv, gamma
import scipy

######################################################
# Basic kernels
######################################################

def matern_kernel(X_a, X_b, hyper_param=[1, 1], nu=1.5):
    """
    Computes the Matern Kernel for inputs X_a and X_b.
    
    X_a : First input array
    X_b : Second input array
    hyper_param : List of hyperparameters [variance, length-scale]
    nu : Smoothness parameter  
    """
    
    # Euclidean distance
    dist = scipy.spatial.distance.cdist(X_a, X_b, metric='euclidean')
    
    # Exponential kernel
    if nu == 0.5:
        return hyper_param[0] * np.exp(-dist / hyper_param[1])
    # Matern 3_2
    elif nu == 1.5:
        factor = np.sqrt(3) * dist / hyper_param[1]
        return hyper_param[0] * (1 + factor) * np.exp(-factor)
    # Matern 5_2
    elif nu == 2.5:
        factor = np.sqrt(5) * dist / hyper_param[1]
        return hyper_param[0] * (1 + factor + (5 * dist ** 2) / (3 * hyper_param[1] ** 2)) * np.exp(-factor)
    # General
    else:
        factor = (np.sqrt(2 * nu) * dist) / hyper_param[1]
        return hyper_param[0] * (2 ** (1 - nu) / gamma(nu)) * (factor ** nu) * kv(nu, factor) if dist > 0 else 1.0


def rbf_kernel(X_a, X_b, hyper_param=[1, 1]):
    """
    Computes the radial basis function kernel for inputs X_a and X_b.
    
    X_a : First input array
    X_b : Second input array
    hyper_param : List of hyperparameters [variance, length-scale]
    """

    # Square Euclidean distance
    sqdist = scipy.spatial.distance.cdist(X_a, X_b, metric='sqeuclidean')
    return hyper_param[0] * np.exp(-0.5 * sqdist / hyper_param[1]**2)


def rat_quad_kernel(X_a, X_b, hyper_param=[1, 1, 1]):
    """
    Computes the rational quadratic kernel for inputs X_a and X_b.
    
    X_a : First input array
    X_b : Second input array
    hyper_param : List of hyperparameters [variance, length-scale, alpha]
    """

    sqdist = scipy.spatial.distance.cdist(X_a, X_b, metric='sqeuclidean')
    return hyper_param[0] * (1.0 + 0.5*sqdist/hyper_param[1]**2/hyper_param[2])**(-1.0*hyper_param[2])


######################################################
# Non-seperable kernels
######################################################

def matern_kernel_NS(X_a, X_b, var, Lambda_inv, nu=1.5):
    """
    Computes the non-separable Matern Kernel for inputs X_a and X_b.
    
    X_a : First input array
    X_b : Second input array
    var : Kernel variance
    Lambda_inv : Inverse of covariance matrix
    nu : Smoothness parameter  
    """
    
    # Mahalanobis distance
    dist = scipy.spatial.distance.mahalanobis(X_a, X_b, Lambda_inv)
    
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
    
    X_a : First input array
    X_b : Second input array
    var : Kernel variance
    Lambda_inv : Inverse of covariance matrix
    """
    
    # Square Mahalanobis distance
    sqdist = scipy.spatial.distance.mahalanobis(X_a, X_b, Lambda_inv)**2
    return var * np.exp(-0.5 * sqdist)


def rational_quadratic_kernel(X_a, X_b, var, Lambda_inv, alpha):
    """
    Computes the non-separable rational quadratic Kernel for inputs X_a and X_b.
    
    X_a : First input array
    X_b : Second input array
    var : Kernel variance
    Lambda_inv : Inverse of covariance matrix
    """

    # Mahalanobis distance
    sqdist = scipy.spatial.distance.mahalanobis(X_a, X_b, Lambda_inv)**2
    return var * (1 + (sqdist / (2 * alpha)))**(-alpha)



######################################################
# Make custom kernels
######################################################

def make_kernel(kern_labels, kern_ops, X_a, X_b, hypers):
    """
    Make kernel from either one or a comnination of kernels for the case of
    1D inputs.
    
    kern_ops : List of kernels to use (RBF, EXP, MATERN_3_2, MATERN_5_2 or RAT_QUAD)
    kern_ops : List of possible kernel operations (either * or +)
    X_a : First input array
    X_b : Second input array
    hyper_param : List of hyperparameters
    """

    # set index for hyperparameters
    idx = 0
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
            
        if i == 0:
            kernel = K_i
        else:
            if kern_ops[i-1] == '*':
                kernel *= K_i
            elif kern_ops[i-1] == '+':
                kernel += K_i
    
    return kernel

def make_kernel_nD(kern_labels, kern_ops, X_a, X_b, hypers):

    """
    Make kernel from either one or a comnination of kernels for the case of
    seperable nD inputs.
    
    kern_ops : List of kernels to use (RBF, EXP, MATERN_3_2, MATERN_5_2 or RAT_QUAD)
    kern_ops : List of possible kernel operations (either * or +)
    X_a : First input array
    X_b : Second input array
    hyper_param : List of hyperparameters
    """

    idx = 0

    # Set kernel
    for i in range(len(kern_labels)):
        X_1 = X_a[:,i]
        X_2 = X_b[:,i]
        if kern_labels[i] == 'RBF':
            K_i = rbf_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2])
            idx += 2
        elif kern_labels[i] == 'EXP':
            K_i = matern_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2], nu=0.5) 
            idx += 2
        elif kern_labels[i] == 'MATERN_3_2':
            K_i = matern_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2], nu=1.5)
            idx += 2 
        elif kern_labels[i] == 'MATERN_5_2':
            K_i = matern_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2], nu=2.5)
            idx += 2
        elif kern_labels[i] == 'RAT_QUAD':
            K_i = rat_quad_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+3])
            idx += 3
            
        if i == 0:
            kernel = K_i
        else:
            if kern_ops[i-1] == '*':
                kernel *= K_i
            elif kern_ops[i-1] == '+':
                kernel += K_i
    return kernel

def make_kernel_NS(kern_labels, kern_ops, X_a, X_b, hypers):

    """
    Make kernel from either one or a comnination of non-seperable kernels.

    kern_ops : List of kernels to use (RBF, EXP, MATERN_3_2, MATERN_5_2 or RAT_QUAD)
    kern_ops : List of possible kernel operations (either * or +)
    X_a : First input array
    X_b : Second input array
    hyper_param : List of hyperparameters
    """

    idx = 0

    # Set kernel
    for i in range(len(kern_labels)):
        X_1 = X_a[:,i]
        X_2 = X_b[:,i]
        if kern_labels[i] == 'RBF':
            K_i = rbf_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2])
            idx += 2
        elif kern_labels[i] == 'EXP':
            K_i = matern_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2], nu=0.5) 
            idx += 2
        elif kern_labels[i] == 'MATERN_3_2':
            K_i = matern_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2], nu=1.5)
            idx += 2 
        elif kern_labels[i] == 'MATERN_5_2':
            K_i = matern_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2], nu=2.5)
            idx += 2
        elif kern_labels[i] == 'RAT_QUAD':
            K_i = rat_quad_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+3])
            idx += 3
            
        if i == 0:
            kernel = K_i
        else:
            if kern_ops[i-1] == '*':
                kernel *= K_i
            elif kern_ops[i-1] == '+':
                kernel += K_i
    return kernel