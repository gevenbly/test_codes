
import numpy as np
from numpy import linalg as LA
import tensornetwork as tn
from tensornetwork.block_sparse import linalg as BLA
from tensornetwork.block_sparse.blocksparse_utils import fuse_charges
from tensornetwork.block_sparse import BlockSparseTensor as BT
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparse_utils import (
    _find_transposed_diagonal_sparse_blocks)

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


#######################################
def orthog_sym(arr, pivot=1):

  arr_size = arr.shape
  arr = arr.reshape([np.prod(arr_size[:pivot]), np.prod(arr_size[pivot:])])
  utemp, _, vtemph = BLA.svd(arr, full_matrices=False)
  return (utemp @ vtemph).reshape(arr_size)


#######################################
def eye_sym(charges, flows, pivot=1):

  ind_temp = [Index(charges[n], flows[n]) for n in range(len(flows))]
  arr = BT.zeros(ind_temp)

  sparse_blocks, cs, dims = _find_transposed_diagonal_sparse_blocks(
      charges, flows, pivot, list(range(len(charges))))
  for k in range(len(cs)):
    arr.data[sparse_blocks[k]] = np.eye(N=dims[0, k], M=dims[1, k]).flatten()
  return arr

