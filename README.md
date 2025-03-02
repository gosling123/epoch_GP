# epoch_GP

Gaussian Process Regression code to use on EPOCH PIC code outputs.
Code uses the assunption of a null mean funvtion, however the code can handle various requests such as:

* **Custom Kernel** - Can define a custom kernel from a combination of defined models.

* **Input warping** - Using Kumaraswamy distribution $F(x, \theta_{iw}) = 1 - (1 - x^a)^b, where a and b are optimsed along with kernel hyperparameters

* **Output Warping** - Using a set of defined transformations. Can be also handle composite output warping (several layers of warping).

* **Heteroscedastic model** - Can train a model with varying noise across the input space.

* **input/output/hyper-paramter optimisation** - Optimised by finding the minimum of the negative of the log marginal likelihood

**To Add** - Kernel gradients, Gradient of log marginal likelihood with respect to input/output/hyper-paramter optimisation

## How To Use The Code

Gaussian process models are stored in the `Model` directory. Example scripts of how to run the code are in the `test_example` directory.

#### Initialising Class

To intialise the Gaussian process, then you need to access the class from either `Model/GP_model.py` or `Model/GP_het_scedat.py` for a constant or varying noise model.

* Homoscedastic Model 
`gp = GP_class(X, y, kern, kern_ops, iw, ow_model)`
* Heteroscedastic Model
`gp = GP_hetscedat_class(X, y, y_var, kern, kern_ops,  kern_var, kern_var_ops, iw, ow_model, ow_noise)`

#### Required variables

* X - Inputs (ndarray)
* y - Outputs (1d array)
* y_var - Variance of outputs (1d array)
* kern -  List of kernels to use (See below)
* kern_ops - List of kernel operations
* kern_var - List of kernels to use (for variance model)
* kern_var_ops - List of kernel operations (for variance model)
* iw - Flag to turn input warping on or off
* ow_model - List of output warpings to perform (see below)
* ow_noise - List of output warpings to perform (for variance model)

#### Kernel Set-up
The can handle a variety of versions of the Radial basis function (`RBF`), Exponential (`EXP`), Matern 3/2, 5/2 (`MATERN_3_2`/`MATERN_5_2`) and Rational quadratic (`RAT_QUAD`) kernels.

The main kind of kernels than can be built are Isotropic, Seperable, Non-seperable and Automatic Relevance Detection (ARD) kernels. A combination of these can be also intialised by using the `kern_ops` variable.

###### Isotropic kernels
Depends only on the Euclidean distance between two input points. Here the length-scale paramter does not change for each input dimension. For the example of intialising an isotropic RBF kernel:

\[
k(\mathbf{x}, \mathbf{x'}) = \sigma^2 \frac{2}{\pi} \sin^{-1} \left(\frac{\mathbf{x}^\top \mathbf{x'}}{\|\mathbf{x}\|\|\mathbf{x'}\|} \right).
\]

* `kern = ['RBF']` - Isotrpoic across all input dimensions
* `kern = ['RBF_ISO_[1,2,3]']` - Isotrpoic across input dimensions 1, 2 and 3


