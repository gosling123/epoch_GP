# epoch_GP

  

Gaussian Process Regression code to use on EPOCH PIC code outputs.

Code uses the assumption of a null mean function, however the code can handle various requests such as:

  

*  **Custom Kernel** - Can define a custom kernel from a combination of defined models.

  

*  **Input warping** - Using Kumaraswamy distribution.
  

*  **Output Warping** - Using a set of defined transformations. Can be also handle composite output warping (several layers of warping).

  

*  **Heteroscedastic model** - Can train a model with varying noise across the input space.

  

*  **input/output/hyper-paramter optimisation** - Optimised by finding the minimum of the negative of the log marginal likelihood

  

**To Add** - Kernel gradients, Gradient of log marginal likelihood with respect to input/output/hyper-paramter optimisation

  

## How To Use The Code

  

Gaussian process models are stored in the `Model` directory. Example scripts of how to run the code are in the `test_example` directory.

  

#### Initialising Class

  

To initialise the Gaussian process, then you need to access the class from either `Model/GP_model.py` or `Model/GP_het_scedat.py` for a constant or varying noise model.

  

* Homoscedastic Model

`gp = GP_class(X, y, kern, kern_ops, iw, ow_model)`

* Heteroscedastic Model

`gp = GP_hetscedat_class(X, y, y_var, kern, kern_ops, kern_var, kern_var_ops, iw, ow_model, ow_noise)`

  

#### Required variables

  

* X - Inputs (ndarray)

* y - Outputs (1d array)

* y_var - Variance of outputs (1d array)

* kern - List of kernels to use (See below)

* kern_ops - List of kernel operations

* kern_var - List of kernels to use (for variance model)

* kern_var_ops - List of kernel operations (for variance model)

* iw - Flag to turn input warping on or off

* ow_model - List of output warping to perform (see below)

* ow_noise - List of output warping to perform (for variance model)

  

### Kernel Set-up

The can handle a variety of versions of the Radial basis function (`RBF`), Exponential (`EXP`), Matern 3/2, 5/2 (`MATERN_3_2`/`MATERN_5_2`) and Rational quadratic (`RAT_QUAD`) kernels.

  

The main kind of kernels than can be built are Isotropic, Separable/Summation, Non-separable and Automatic Relevance Detection (ARD) kernels. A combination of these can be also initialised by using the `kern_ops` variable.

#### Isotropic kernels

Depends only on the Euclidean distance between two input points. Here the length-scale parameter does not change for each input dimension. For the example of initialising an isotropic RBF kernel:
$$
k(\mathbf{x}, \mathbf{x'}) = \sigma^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x'}\|^2}{2\ell^2}\right)
$$
*  `kern = ['RBF']` - Isotropic across all input dimensions

*  `kern = ['RBF_ISO_[1,2,3]']` - Isotropic across input dimensions 1, 2 and 3 only

#### Separable Kernels
Kernels can be written as the product of one-dimensional kernel functions acting on individual input dimensions
$$
k(\mathbf{x}, \mathbf{x'}) = \prod_{i=1}^{d} k_i(x_i, x'_i).
$$
For example,  to initialise a separable for 3D inputs 
 
 *  `kern = ['RBF_[1]', MATERN_3_2_[2], MATERN_5_2_[3]] ; kern_ops = ['*', '*']` 
 Here the first dimensions is defined by a RBF kernel and the others by a variation of the MATERN kernel.

#### General product kernels
A kernel which uses multiple dimensions can be defined for example for a 2D model
* `kern = ['RBF_[1], RBF_[2], RBF_[1, 2]] ; kern_ops = ['*', '*']`  
where the last kernel defines a RBF kernel where `X` is only defined in the first input dimension and `X'` on the second.

#### Summation Kernels
Kernels can be written as the summation of other kernels.
$$
k(\mathbf{x}, \mathbf{x'}) = k_1(\mathbf{x}, \mathbf{x'}) + k_2(\mathbf{x}, \mathbf{x'}) + \dots + k_n(\mathbf{x}, \mathbf{x'}).
$$
This can be done using `kern_ops = ['+', '+', ...]`.

#### Automatic Relevance Detection Kernels
Automatic Relevance Detection (ARD) modifies kernel functions to include separate length scales for each input dimension, allowing the model to infer the relevance of each feature. The ARD squared exponential kernel is:
$$
k(\mathbf{x}, \mathbf{x'}) = \sigma^2 \exp\left(-\sum_{i=1}^{d} \frac{(x_i - x'_i)^2}{2\ell_i^2}\right).
$$
If a length scale  $\ell_i$ is large, the corresponding feature is less relevant, whereas a small $\ell_i$ suggests high relevance.

To initialise the above ARD kernel we do :

*  `kern = ['RBF_ARD']` - ARD across all input dimensions

*  `kern = ['RBF_ARD_[1,2,3]']` - Isotropic across input dimensions 1, 2 and 3 only

####  Non-Separable Kernels (Full Covariance Matrix)
A  non-separable kernel  based on the Mahalanobis distance  captures dependencies between input dimensions through a covariance matrix. The Mahalanobis distance is defined as:
$$
d_M(x,x')=\sqrt{(x - x')^T \Lambda^{-1} (x - x')}
$$

where:
* $\Lambda$ is the covariance matrix (positive definite).
* $\Lambda^{-1}$ accounts for feature correlations.

This distance generalises the Euclidean distance by considering correlations between features, making it highly useful for non-isotropic (directionally dependent) data.

For the RBF, this is now given by:
$$
k_{RBF}​(x,x′)=\sigma^2\exp\left((x - x')^T \Lambda^{-1} (x - x')\right),
$$
which in the code is given by:

*  `kern = ['RBF_NS']` - ARD across all input dimensions

*  `kern = ['RBF_NS_[1,2,3]']` - Isotropic across input dimensions 1, 2 and 3 only

When optimising the non-separable case, we assume $\Lambda = LL^{T}$. That way we only find the optimal lower triangular elements when optimising and then use $L$ to define the covariance.

#### Combinations
Combinations of the above are allowed using the `kern_ops` variable.

However:
* All input dimensions have at least been used once. Either by none input labels (as this assumes all are to be used), or by having defined each input dimension in the number labelling.
* Non-separable, and ARD flags cannot be used for 1D inputs.