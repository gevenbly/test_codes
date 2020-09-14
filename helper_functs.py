
import numpy as np
from numpy import linalg as LA
import tensornetwork as tn

#######################################
def orthog(arr, pivot=1):

  arr_size = arr.shape
  arr = arr.reshape(np.prod(arr_size[:pivot]), np.prod(arr_size[pivot:]))
  utemp, stemp, vtemph = LA.svd(arr, full_matrices=False)
  return (-utemp @ vtemph).reshape(arr_size)


#######################################
def trunct_eigh(rho, chi=None):

  rho_sz = rho.shape
  n_inds = rho.ndim // 2
  rho = rho.reshape(np.prod(rho_sz[:n_inds]), np.prod(rho_sz[:n_inds]))

  if chi is None:
    chi = rho.shape[0]
  chi = min(rho.shape[0], chi)

  dtemp, utemp = LA.eigh(rho)
  dtemp = np.flipud(dtemp)
  utemp = np.fliplr(utemp)

  return dtemp[:chi], utemp[:, :chi].reshape(*rho_sz[:n_inds], chi)


#######################################
def expand_dims(arr, new_dims):

  tensors = [0] * (arr.ndim + 1)
  tensors[0] = arr
  connects = [0] * (arr.ndim + 1)
  connects[0] = list(range(1, 1 + arr.ndim))
  for k in range(arr.ndim):
    tensors[k + 1] = np.eye(arr.shape[k], new_dims[k])
    connects[k + 1] = [(k + 1), -(k + 1)]

  return tn.ncon(tensors, connects)
