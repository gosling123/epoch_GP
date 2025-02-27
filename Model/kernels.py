import numpy as np
import scipy.spatial
from scipy.special import kv, gamma
import scipy
import sys


######################################################
# Basic kernels
######################################################

def matern_kernel(X_a, X_b, hyper_param=[1, 1], nu=1.5):
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

    Returns:
    ndarray
        The computed Matern kernel matrix.
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
    
    Parameters:
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    hyper_param : list, optional
        List of hyperparameters [variance, length-scale]. Default is [1, 1].

    Returns:
    ndarray
        The computed RBF kernel matrix.
    """
    
    # Square Euclidean distance
    sqdist = scipy.spatial.distance.cdist(X_a, X_b, metric='sqeuclidean')
    return hyper_param[0] * np.exp(-0.5 * sqdist / hyper_param[1]**2)


def rat_quad_kernel(X_a, X_b, hyper_param=[1, 1, 1]):
    """
    Computes the rational quadratic kernel for inputs X_a and X_b.
    
    Parameters:
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    hyper_param : list, optional
        List of hyperparameters [variance, length-scale, alpha]. Default is [1, 1, 1].

    Returns:
    ndarray
        The computed rational quadratic kernel matrix.
    """
    
    sqdist = scipy.spatial.distance.cdist(X_a, X_b, metric='sqeuclidean')
    return hyper_param[0] * (1.0 + 0.5*sqdist/hyper_param[1]**2/hyper_param[2])**(-1.0*hyper_param[2])


######################################################
# Non-separable kernels
######################################################

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
    sqdist = scipy.spatial.distance.mahalanobis(X_a, X_b, Lambda_inv)**2
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
    sqdist = scipy.spatial.distance.mahalanobis(X_a, X_b, Lambda_inv)**2
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
    kern_ops : list of str
        List of kernel operations ('+' or '*').
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
    Make kernel from either one or a combination of kernels for the case of
    separable nD inputs.
    
    Parameters:
    kern_labels : list of str
        List of kernel labels ('RBF', 'EXP', 'MATERN_3_2', 'MATERN_5_2', 'RAT_QUAD').
    kern_ops : list of str
        List of kernel operations ('+' or '*').
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
    Make kernel from either one or a combination of non-separable kernels.

    Parameters:
    kern_labels : list of str
        List of kernel labels ('RBF', 'EXP', 'MATERN_3_2', 'MATERN_5_2', 'RAT_QUAD').
    kern_ops : list of str
        List of kernel operations ('+' or '*').
    X_a : ndarray
        First input array.
    X_b : ndarray
        Second input array.
    hypers : list
        List of hyperparameters for the kernels.

    Returns:
    ndarray
        The resulting combined non-separable kernel matrix.
    """
    
    
    if len(kern_labels) == 1:
        # Which kernel
        label = kern_labels[0].split("_NS")[0]

        # Fill lower triangular matrix
        L = np.zeros((X_a.shape[-1], X_a.shape[-1]))
        if label == 'RAT_QUAD':
            L[np.tril_indices(X_a.shape[-1])] = hypers[1:-1]
        else:
            L[np.tril_indices(X_a.shape[-1])] = hypers[1:] 
        
        # Compute inverse covariance
        Lambda = np.dot(L, L.T)
        Lambda_inv = np.linalg.inv(Lambda)

        if label == 'RBF':
            kernel = rbf_kernel_NS(X_a, X_b, hypers[0], Lambda_inv)
        elif label == 'EXP':
            kernel = matern_kernel_NS(X_a, X_b, hypers[0], Lambda_inv, nu=0.5)
        elif label == 'MATERN_3_2':
            kernel = matern_kernel_NS(X_a, X_b, hypers[0], Lambda_inv, nu=1.5)
        elif label == 'MATERN_5_2':
            kernel = matern_kernel_NS(X_a, X_b, hypers[0], Lambda_inv, nu=2.5)
        elif label == 'RAT_QUAD':
            kernel = rat_quad_kernel_NS(X_a, X_b, hypers[0], Lambda_inv, hypers[-1])
    
    else:
        idx = 0
        # Set kernel
        for i in range(len(kern_labels)):
            # Check if non-seperable or not        
            check = kern_labels[i].split("_NS")[0] if "_NS" in kern_labels else None
            
            # Seperable
            if check == None:
                dim = extract_numbers_after_kernel(kern_labels[i])
                if len(dim) != 1:
                    sys.exit('(ERROR): If using a mix of seperbale and non-sperable, then make sure seperable only has one number after the label, and non-seperable have two with lowest dimension first')
                else:
                    X_1 = X_a[:,int(dim)-1]
                    X_2 = X_b[:,int(dim)-1]
                    if kern_labels[i][:-2] == 'RBF':
                        K_i = rbf_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2])
                        idx += 2
                    elif kern_labels[i][:-2] == 'EXP':
                        K_i = matern_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2], nu=0.5) 
                        idx += 2
                    elif kern_labels[i][:-2] == 'MATERN_3_2':
                        K_i = matern_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2], nu=1.5)
                        idx += 2 
                    elif kern_labels[i][:-2] == 'MATERN_5_2':
                        K_i = matern_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+2], nu=2.5)
                        idx += 2
                    elif kern_labels[i][:-2] == 'RAT_QUAD':
                        K_i = rat_quad_kernel(X_1[:,None], X_2[:,None], hyper_param=hypers[idx:idx+3])
                        idx += 3
            # Non seperable
            else:
                dim = extract_numbers_after_kernel(kern_labels[i])
                if len(dim) > 2 and dim[0] > dim[1]:
                    sys.exit('(ERROR): If using a mix of seperbale and non-sperable, then make sure seperable only has one number after the label, and non-seperable have two with lowest dimension first (i.e ascending order)')
                else:
                    
                    X_1 = X_a[:, [int(dim[0])-1, int(dim[1])-1]]
                    X_2 = X_b[:, [int(dim[0])-1, int(dim[1])-1]]
                    
                    # Which kernel
                    label = kern_labels[i].split("_NS")[0]

                    Ncov_params = int(0.5*X_1.shape[-1]*(X_1.shape[-1]+1))
                    # Fill lower triangular matrix
                    L = np.zeros((X_1.shape[-1], X_1.shape[-1]))
                    L[np.tril_indices(X_1.shape[-1])] = hypers[idx+1:int(0.5*X_1.shape[-1]*(X_1.shape[-1]+1))+1]
        
                    # Compute inverse covariance
                    Lambda = np.dot(L, L.T)
                    Lambda_inv = np.linalg.inv(Lambda)

                if label == 'RBF':
                    K_i = rbf_kernel_NS(X_1, X_2, hypers[idx], Lambda_inv)
                    idx += int(X_1.shape[-1]*(X_1.shape[-1]+1)/2) + 1 
                elif label == 'EXP':
                    K_i = matern_kernel_NS(X_1, X_2, hypers[idx], Lambda_inv, nu=0.5)
                    idx += int(X_1.shape[-1]*(X_1.shape[-1]+1)/2) + 1 
                elif label == 'MATERN_3_2':
                    K_i = matern_kernel_NS(X_1, X_2, hypers[idx], Lambda_inv, nu=1.5)
                    idx += int(X_1.shape[-1]*(X_1.shape[-1]+1)/2) + 1 
                elif label == 'MATERN_5_2':
                    K_i = matern_kernel_NS(X_1, X_2, hypers[idx], Lambda_inv, nu=2.5)
                    idx += int(X_1.shape[-1]*(X_1.shape[-1]+1)/2) + 1
                elif label == 'RAT_QUAD':
                    K_i = rat_quad_kernel_NS(X_1, X_2, hypers[idx], Lambda_inv, hypers[int(0.5*X_1.shape[-1]*(X_1.shape[-1]+1))+1])
                    idx += int(X_1.shape[-1]*(X_1.shape[-1]+1)/2) + 2    

            if i == 0:
                kernel = K_i
            else:
                if kern_ops[i-1] == '*':
                    kernel *= K_i
                elif kern_ops[i-1] == '+':
                    kernel += K_i
    
    return kernel

def extract_numbers_after_kernel(s):
    kernels = ('RBF', 'EXP', 'MATERN_3_2', 'MATERN_5_2', 'RAT_QUAD')
    
    for kernel in kernels:
        if s.startswith(kernel):  # Check if string starts with a known kernel
            num_part = "".join(char for char in s[len(kernel):] if char.isdigit())
            return int(num_part) if num_part else None  # Convert to int if found
    
    return None  # Return None if no kernel is matched
