# EPOCH_GP

Gaussian Process Regression code to use on EPOCH PIC code outputs.

Code uses the assumption of a null mean function, however, the code can handle various requests such as:

* **Custom Kernel** - Can define a custom kernel from a combination of defined models.
* **Input warping** - Using Kumaraswamy distribution.
* **Output Warping** - Using a set of defined transformations. Can also handle composite output warping (several layers of warping).
* **Heteroscedastic model** - Can train a model with varying noise across the input space.
* **Input/Output/Hyper-parameter optimisation** - Optimised by finding the minimum of the negative of the log marginal likelihood.

**To Add** - Kernel gradients, Gradient of log marginal likelihood with respect to input/output/hyper-parameter optimisation.

## How To Use The Code

Gaussian process models are stored in the `Model` directory. Example scripts of how to run the code are in the `test_example` directory.

### Initialising Class

To initialise the Gaussian process, you need to access the class from either `Model/GP_model.py` or `Model/GP_het_scedat.py` for a constant or varying noise model.

#### Homoscedastic Model
```python
gp = GP_class(X, y, kern, kern_ops, iw, ow_model)
```
#### Heteroscedastic Model
```python
gp = GP_hetscedat_class(X, y, y_var, kern, kern_ops, kern_var, kern_var_ops, iw, ow_model, ow_noise)
```

### Required Variables

* `X` - Inputs (ndarray)
* `y` - Outputs (1d array)
* `y_var` - Variance of outputs (1d array)
* `kern` - List of kernels to use (See below)
* `kern_ops` - List of kernel operations
* `kern_var` - List of kernels to use (for variance model)
* `kern_var_ops` - List of kernel operations (for variance model)
* `iw` - Flag to turn input warping on or off
* `ow_model` - List of output warping to perform (see below)
* `ow_noise` - List of output warping to perform (for variance model)

## Kernel Set-up

The code can handle a variety of versions of the Radial Basis Function (`RBF`), Exponential (`EXP`), Matern 3/2, 5/2 (`MATERN_3_2`/`MATERN_5_2`), and Rational Quadratic (`RAT_QUAD`) kernels.

The main types of kernels that can be built are Isotropic, Separable/Summation, Non-separable, and Automatic Relevance Detection (ARD) kernels. A combination of these can also be initialised using the `kern_ops` variable.

### Isotropic Kernels

Isotropic kernels depend only on the Euclidean distance between two input points. The length-scale parameter does not change for each input dimension.

For an isotropic RBF kernel:
```math
k(\mathbf{x}, \mathbf{x'}) = \sigma^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x'}\|^2}{2\ell^2}\right)
```
* `kern = ['RBF']` - Isotropic across all input dimensions.
* `kern = ['RBF_ISO_[1,2,3]']` - Isotropic across input dimensions 1, 2, and 3 only.

### Separable Kernels

Separable kernels can be written as the product of one-dimensional kernel functions acting on individual input dimensions:
```math
k(\mathbf{x}, \mathbf{x'}) = \prod_{i=1}^{d} k_i(x_i, x'_i)
```
Example for 3D inputs:
```python
kern = ['RBF_[1]', 'MATERN_3_2_[2]', 'MATERN_5_2_[3]']
kern_ops = ['*', '*']
```

### General Product Kernels

A kernel using multiple dimensions for a 2D model:
```python
kern = ['RBF_[1]', 'RBF_[2]', 'RBF_[1,2]']
kern_ops = ['*', '*']
```
where the last kernel defines an RBF kernel where `X` is only defined in the first input dimension and `X'` on the second.

### Summation Kernels

Summation kernels combine multiple kernels:
```math
k(\mathbf{x}, \mathbf{x'}) = k_1(\mathbf{x}, \mathbf{x'}) + k_2(\mathbf{x}, \mathbf{x'}) + \dots + k_n(\mathbf{x}, \mathbf{x'})
```
Use `kern_ops = ['+', '+', ...]` to define summation operations.

### Automatic Relevance Detection (ARD) Kernels

ARD kernels introduce separate length scales for each input dimension:
```math
k(\mathbf{x}, \mathbf{x'}) = \sigma^2 \exp\left(-\sum_{i=1}^{d} \frac{(x_i - x'_i)^2}{2\ell_i^2}\right)
```
* `kern = ['RBF_ARD']` - ARD across all input dimensions.
* `kern = ['RBF_ARD_[1,2,3]']` - ARD across input dimensions 1, 2, and 3 only.

### Non-Separable Kernels (Full Covariance Matrix)

A non-separable kernel based on the Mahalanobis distance:
```math
d_M(x,x')=\sqrt{(x - x')^T \Lambda^{-1} (x - x')}
```
where `\Lambda` is the covariance matrix (positive definite), and `\Lambda^{-1}` accounts for feature correlations.

For the RBF kernel:
```math
k_{RBF}(x,x′)=\sigma^2\exp\left((x - x')^T \Lambda^{-1} (x - x')\right)
```
Defined in code as:
* `kern = ['RBF_NS']` - Non-separable across all input dimensions.
* `kern = ['RBF_NS_[1,2,3]']` - Non-separable across input dimensions 1, 2, and 3 only.

During optimisation, we assume `\Lambda = LL^{T}`, so only the lower triangular elements are optimised.

### Combinations

Combinations of kernels are allowed using the `kern_ops` variable, with the following rules:
* All input dimensions must be used at least once.
* Non-separable and ARD flags cannot be used for 1D inputs.

