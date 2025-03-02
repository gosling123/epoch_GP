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

## Input Warping

Gaussian processes assume a stationary covariance function, but real-world functions often vary across input space. Input warping adapts to such variations. This implementation uses the Kumaraswamy distribution:

```math
F(x, \theta_{iw}) = 1 - (1 - x^a)^b,
```

where `a` and `b` are transformation parameters, learned alongside kernel hyperparameters. Inputs must be between `[0,1]`, so we first apply an affine transformation:

```math
\tilde{x} = \frac{x - x_{min}}{x_{max} - x_{min}} + \epsilon,
```

where `\epsilon` ensures numerical stability and allows predictions slightly beyond the training domain.

## Output Warping

Standard Gaussian processes assume Gaussian-distributed outputs, which may not always hold. Output warping transforms outputs to better fit this assumption. Common transformations include:



### Composite Output Warping

For high-dimensional problems, a single transformation may not be sufficient. Composite warping applies multiple transformations in sequence:

```math
\tilde{y} = \phi_n(\phi_{n-1}(\phi_{n-2}(...), \theta_{n-1}), \theta_n).
```

Affine transformations are typically included to standardize the output to zero mean and unit variance. If a nonzero mean function is used, the Affine transform parameters can be adjusted accordingly.

## Input Warping

Gaussian processes assume a stationary covariance function, but real-world functions often vary across input space. Input warping adapts to such variations. This implementation uses the Kumaraswamy distribution:

```math
F(x, \theta_{iw}) = 1 - (1 - x^a)^b,
```

where `a` and `b` are transformation parameters, learned alongside kernel hyperparameters. Inputs must be between `[0,1]`, so we first apply an affine transformation:

```math
\tilde{x} = \frac{x - x_{min}}{x_{max} - x_{min}} + \epsilon,
```

where `\epsilon` ensures numerical stability and allows predictions slightly beyond the training domain.

## Output Warping

Standard Gaussian processes assume Gaussian-distributed outputs, which may not always hold. Output warping transforms outputs to better fit this assumption. Common transformations include:

- **Affine Transformation**: Scales and shifts the output:
  ```math
  \phi(y) = a y + b
  ```
  Inverse: 
  ```math
  \phi^{-1}(\tilde{y}) = \frac{\tilde{y} - b}{a}
  ```
  Jacobian: 
  ```math
  \frac{d\phi}{dy} = a
  ```

- **Logarithmic Transformation**: Useful for positive outputs:
  ```math
  \phi(y) = \ln(y)
  ```
  Inverse: 
  ```math
  \phi^{-1}(\tilde{y}) = \exp(\tilde{y})
  ```
  Jacobian: 
  ```math
  \frac{d\phi}{dy} = \frac{1}{y}
  ```

- **Box-Cox Transformation**: Handles skewed data:
  ```math
  \phi(y) = \frac{\text{sgn}(y)|y|^{a-1} - 1}{a-1}
  ```
  Inverse: 
  ```math
  \phi^{-1}(\tilde{y}) = \text{sgn}((a-1)\tilde{y} +1) |(a-1)\tilde{y} + 1|^{1/(a-1)}
  ```
  Jacobian: 
  ```math
  \frac{d\phi}{dy} = (a-1)|y|^{a-2}
  ```

- **Sinh-Arcsinh Transformation**: Adjusts tail behavior:
  ```math
  \phi(y) = \sinh(b\sinh^{-1}y -a)
  ```
  Inverse: 
  ```math
  \phi^{-1}(\tilde{y}) = \sinh\left(\frac{\sinh^{-1}(\tilde{y}) - a}{b} \right)
  ```
  Jacobian: 
  ```math
  \frac{d\phi}{dy} = \frac{b \cosh(b \sinh^{-1}y -a)}{\sqrt{1 + y^2}}
  ```

### Composite Output Warping

For high-dimensional problems, a single transformation may not be sufficient. Composite warping applies multiple transformations in sequence:

```math
\tilde{y} = \phi_n(\phi_{n-1}(\phi_{n-2}(...), \theta_{n-1}), \theta_n).
```

Affine transformations are typically included to standardize the output to zero mean and unit variance. If a nonzero mean function is used, the Affine transform parameters can be adjusted accordingly.

#### GP Inference

Given the training data \( (X, y) \), the posterior mean and variance at test points \( X^* \) are computed as follows:

- **Posterior Mean**:

    ```math
  \mu(\tilde{X}^*) = \mathbf{K}(\tilde{X}^*, \tilde{X})\left[\mathbf{K}(\tilde{X}, \tilde{X}) + \sigma^2_{n}\mathbf{I}\right]^{-1}\mathbf{\tilde{y}}
  ```

- **Posterior Variance**:

    ```math
  \text{Var}(\tilde{X}^*) = \mathbf{K}(\tilde{X}^*, \tilde{X}) - \mathbf{K}(\tilde{X}^*, \tilde{X})\left[\mathbf{K}(\tilde{X}, \tilde{X}) + \sigma^2_{n}\mathbf{I}\right]^{-1}\mathbf{K}(\tilde{X}, \tilde{X}^*)
  ```

Here, `K` denotes the kernel matrix, and `\tilde{X} = F(X, \theta_{iw})` and ` \tilde{y} = \phi(y, \theta_{ow})`. 

To improve numerical stability, Cholesky decomposition is applied to avoid matrix inversion:

```math
\mathbf{L}\mathbf{L}^{T} = \mathbf{K}(\tilde{X}, \tilde{X}) + \sigma_n^2\mathbf{I}
```

where \( \mathbf{L} \) is a lower triangular matrix. From this, we define the weights \( \alpha \) as:

  ```math
\mathbf{\alpha} = \left[\mathbf{K}(\tilde{X}, \tilde{X}) + \sigma_n^2\mathbf{I}\right]^{-1} \tilde{\mathbf{y}} \quad \rightarrow \quad \alpha = \mathbf{L}^T \backslash (\mathbf{L} \backslash \mathbf{y})
```

Thus, the mean becomes:

  ```math
\mu(X^{\star}) = \mathbf{K}(\tilde{X}^*, \tilde{X})\mathbf{\alpha}
```

Similarly, for variance:

  ```math
\mathbf{v} = \mathbf{L} \backslash \mathbf{K}(\tilde{X}, \tilde{X}^*) \quad \rightarrow \quad \text{Var}(\tilde{X}^*) = \mathbf{K}(\tilde{X}^*, \tilde{X}^*) - \mathbf{v}^T\mathbf{v}
```

Finally, the predictions are converted back to the original output space (non-warped), using Gaussian quadrature to approximate the integral:

  ```math
\mathbb{E}\left[y^n\right] = \int_{-\infty}^{\infty} (\phi^{-1}(\tilde{y}))^{n}f_{\tilde{y}}(\tilde{y}) \mathrm{d}\tilde{y}
```

where $w_i$ and \( \beta_i \) are the Gaussian quadrature weights and nodes, respectively. The mean and variance are computed by evaluating \( \mathbb{E}\left[y\right] \) and \( \mathbb{E}\left[y^2\right] - \mathbb{E}\left[y\right]^2 \).

---

#### Log Marginal Likelihood Optimization

The kernel hyperparameters are optimized by maximizing the log marginal likelihood, which is computationally done by minimizing the negative log marginal likelihood. To ensure the optimization is done in the original space (even when using warped data), we perform a change of variables on the probability density:

  ```math
f_{y}(y) = f_{\tilde{y}}(\tilde{y}) \left|\frac{\mathrm{d}\tilde{y}}{\mathrm{d}y}\right|
```

The log marginal likelihood is then defined as:

  ```math
\log \mathbb{P}(y | X) = -\frac{1}{2} \tilde{\mathbf{y}}^T \left[ \mathbf{K}(\tilde{X}, \tilde{X}) + \sigma^2_{n} \mathbf{I} \right]^{-1} \tilde{\mathbf{y}} 
- \frac{1}{2} \log \left|\left[ \mathbf{K}(\tilde{X}, \tilde{X}) + \sigma^2_{n} \mathbf{I} \right]^{-1}\right| 
- \frac{n}{2} \log 2\pi 
+ \sum_{i=1}^{n} \log{\left(\left|\frac{\mathrm{d}\tilde{y}}{\mathrm{d}y}\right|\right)}
```

This optimization is typically performed using `scipy.minimize` with the L-BFGS-B method or `scipy.optimize.differential_evolution` for stochastic global optimization. The differential evolution method requires more function evaluations and is better suited for problems with fewer hyperparameters. For the `minimize` function, multiple restarts are recommended to avoid local minima. Performance can be improved by defining the derivative of the log marginal likelihood, though this is not yet implemented.

--- 

## References

- [Input-Warping](https://arxiv.org/pdf/1402.0929)
- [Output-Warping](https://arxiv.org/abs/1906.09665>)
- [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [scipy.optimize.differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
