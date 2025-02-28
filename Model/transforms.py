#!/bin/python3

import numpy as np
import sys


##################################################################
# Input Warping
##################################################################

class kumaraswamy:
  def __init__(self,a,b):
    """
    Input warping with kumaraswamy distribution
    CDF = 1 - (1 - x^a)^b
    
    a : Transform parameter
    b : Transform parameter
    """
    # Parameters
    self.a = a
    self.b = b
    try:
      if not self.a > 0.0:
        raise Exception('Parameter a must be positive')
      if not self.b > 0.0:
        raise Exception('Parameter b must be positive')
    except:
      pass

  def transform(self,x):
    return 1 - np.power(1-np.power(x,self.a),self.b)
  def inverse(self,x):
    return np.power(1-np.power(1-x,1/self.b),1/self.a)
  def Jacobian(self,x):
    return self.a*self.b*np.power(x,self.a-1)*np.power(1-np.power(x,self.a),self.b-1)
  
  
##################################################################
# Output Warping
##################################################################
class nat_log:
  def __init__(self):
    """
    Transform using natural logarithm
    """
    pass
  def transform(self,y):
    return np.log(y)
  def inverse(self,y):
    return np.exp(y)
  def Jacobian(self,y):
    return 1/y

class sinharcsinh:
  def __init__(self,a,b):
    """
    Output warping Sinh-Arcsinh
    
    Transform:
    f(y) = sinh(b asinh(y) - a)
    
    Inverse:
    f^(-1)(y') = sinh(asinh(y' + a) / b) 
    
    Jacobian:
    dy'/dy = b cosh(b asinh(y) - a) / sqrt(1 + y^2)
    
    a : Transform parameter
    b : Transform parameter
    """
    self.a = a
    self.b = b
    try:
      if not self.b > 0.0:
        raise Exception('Parameter b must be positive')
    except:
      pass
  def transform(self,y):
    return np.sinh(self.b*np.arcsinh(y)-self.a)
  def inverse(self,y):
    return np.sinh((np.arcsinh(y)+self.a)/self.b)
  def Jacobian(self,y):
    return self.b*np.cosh(self.b*np.arcsinh(y)-self.a)/np.sqrt(1+np.power(y,2))

class affine:
  def __init__(self,a,b):
    """
    Output warping Affine
    
    Transform:
    f(y) = a + by
    
    Inverse:
    f^(-1)(y') = (y' - a) / b
    
    Jacobian:
    dy'/dy = b
    
    a : Transform parameter
    b : Transform parameter
    """
    self.a = a
    self.b = b
    try:
      if not self.b > 0.0:
        raise Exception('Parameter b must be positive')
    except:
      pass
  def transform(self,y):
    return self.a + self.b*y
  def inverse(self,y):
    return (y-self.a)/self.b
  def Jacobian(self,y):
    return self.b*np.ones_like(y)

class meanstd(affine):
  """
  Affine transform for mean 0 and unit variance
  """
  def __init__(self, y):
    self.a = -np.mean(y)/np.std(y)
    self.b = 1/np.std(y)

class zero_mean(affine):
  """
  Affine transform for mean 0
  """
  def __init__(self, b, y):
    self.b = b
    self.a = -self.b*np.mean(y)

class unit_var(affine):
  """
  Affine transform for mean 0
  """
  def __init__(self, a, y):
    self.a = a
    self.b = 1.0/np.std(y)
  
  
class boxcox:
  def __init__(self,a):
    """
    Output warping boxcox 
    
    Transform:
    f(y) = (sgn(y) |y|^(a - 1) - 1) / (a - 1)
    
    Inverse:
    f^(-1)(y') = sgn((a - 1) * y' + 1) / |(a - 1) * y' + 1|^{a-1}
    
    Jacobian:
    dy'/dy = (a -1) * |y|^(a-2)
    
    a : Transform parameter
    """
    self.a = a
  def transform(self,y):
    return (np.sign(y)*np.power(np.abs(y),self.a-1)-1)/(self.a-1)
  def inverse(self,y):
    return np.sign((self.a-1)*y + 1)*np.power(np.abs((self.a-1)*y + 1), 1/(self.a-1))
  def Jacobian(self,y):
    return (self.a-1)*np.power(np.abs(y),self.a-2)
  
##################################################################
# General Transforms
##################################################################
    
class zero_one_scale:
  def __init__(self, x_min, x_max, eps=0.01):
    """
    Transform to be between 0 and 1 with buffer
    to aid prediction out of range
      
    eps : buffer factor
    """
    self.eps = eps
    self.x_min = x_min
    self.x_max = x_max
  
  def transform(self, x):
    return ((x - self.x_min) * (1.0 - 2.0*self.eps)) / (self.x_max - self.x_min) + self.eps 
  def inverse(self, x):
    return self.x_min + (x-self.eps) * (self.x_max - self.x_min) / (1.0 - 2.0*self.eps)
  def Jacobian(self, x):
    return (1.0 - 2.0 * self.eps) / (self.x_max - self.x_min) * np.ones_like(x)



##################################################################
# Composite output warpings
##################################################################


def check_ow(warpings, warpings_list):

    invalid_values = [value for value in warpings if value not in warpings_list]
    if invalid_values:
        sys.exit(f"(ERROR): Invalid warping name found: {invalid_values}, please ensure all warpings are in the following list {warpings_list}") 

class output_warp:

  def __init__(self, warpings, params, y):

    warp_labels = ['affine','nat_log', 'boxcox', 'sinharcsinh', 'meanstd',\
                   'zero_mean', 'unit_var']

    self.warping_names = warpings
    self.params = params
    check_ow(self.warping_names, warp_labels)


  def transform(self, y):

    self.warpings = []
    y_warped = y
    idx = 0
    for i,n in enumerate(self.warping_names):

      if n == 'affine':
        self.warpings.append(affine(a=self.params[idx], b=self.params[idx+1]))
        idx += 2
       
      elif n == 'nat_log':
        self.warpings.append(nat_log())

      elif n == 'boxcox':
        self.warpings.append(boxcox(a=self.params[idx]))
        idx += 1
      
      elif n == 'sinharcsinh':
        self.warpings.append(sinharcsinh(a=self.params[idx], b=self.params[idx+1]))
        idx += 2

      elif n == 'meanstd':
        self.warpings.append(meanstd(y=y_warped))
      
      elif n == 'zero_mean':
        self.warpings.append(zero_mean(b=self.params[idx], y=y_warped))
        idx += 1
      
      elif n == 'unit_var':
        self.warpings.append(unit_var(a=self.params[idx], y=y_warped))
        idx += 1

      y_warped = self.warpings[i].transform(y_warped)

    return y_warped

  def inverse(self, y):

    # Assumption here is that a transform has already been called
    y_iwarped = y
    for w in reversed(self.warpings):
      y_iwarped = w.inverse(y_iwarped)

    return y_iwarped 

  def Jacobian(self, y):
    y_warped = y
    jac = np.ones_like(y)
    for w in self.warpings:
      jac *= w.Jacobian(y_warped)
      y_warped = w.transform(y_warped)
    return jac




