# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 08:43:28 2020

@author: gevenbly3
"""

from scipy.sparse.linalg import eigsh
from numpy import linalg as LA
import numpy as np

import tensornetwork as tn

# --------------- Network-0 --------------- # 
# Leading order cost:(chi^4)*(chi_p^5)
chi = 8
chi_p = 6
u = np.random.rand(chi,chi,chi_p,chi_p)
w = np.random.rand(chi_p,chi_p,chi)
ham = np.random.rand(chi,chi,chi,chi,chi,chi)
rho = np.random.rand(chi,chi,chi,chi,chi,chi)

# TTv1.0.5.0P$;@<EHE<?H?@TE`ET?`?B09<963BH9T9N3B`9l9f3>HQTQ`QHKTK`K@<WHW<]H]@
# TW`WT]`]B0c<c6iBHcTcNiB`clcfiF6'N'f'6-N-f-')++))+++''+''''++++(((LV,HQ^LO=]
# 'IQ^:HUYqHQ^VJ$
tensors = [u,u,w,w,w,ham,u.conj(),u.conj(),w.conj(),w.conj(),w.conj(),rho]
connects = [[1,3,10,11],[4,7,12,13],[8,10,17],[11,12,18],[13,14,19],
	[2,5,6,3,4,7],[1,2,9,23],[5,6,16,15],[8,9,22],[23,16,21],[15,14,20],
	[22,21,20,17,18,19]]
con_order = [4,7,5,6,8,17,22,14,20,19,11,23,3,12,1,2,16,10,18,13,15,9,21]
# T_out = tn.ncon(tensors,connects,con_order)

which_env = 2
###########################################
num_tensors = len(tensors)

connect_list = [np.asarray(connects[ele]) for ele in range(num_tensors)]
connect_flat = np.concatenate(connect_list)

if which_env is not None:
  if (which_env < 1) or (which_env > num_tensors):
    raise ValueError(('Parameter `which_env` is out of range: '
                      'should be between 1 and %i') % (num_tensors))
  if min(connect_flat) < 0:
    raise ValueError('Non-trivial `which_env` can only be set for closed '
                     'tensor networks (i.e. no negative indices)')

# generate the set of contraction tree nodes

# remove partial traces from connects and con_order
new_con_order = con_order
for tensor, labs in enumerate(connect_list):
  num_trace = len(labs) - len(np.unique(labs))
  if num_trace > 0:
    
    
    new_con_order = np.delete(new_con_order, np.intersect1d(new_con_order, cont_ind, return_indices=True)[1])

bin_labels = 2 ** np.asarray(range(num_tensors))



    
connect_free = connect_list.pop(which_env-1)
for ind, val in enumerate(connect_free):
  for tensor, labs in enumerate(connect_list):
    connect_list[tensor][labs == val] = -ind - 1
      

 

def partial_trace(A, A_label):
  """ Partial trace on tensor A over repeated labels in A_label """

  num_cont = len(A_label) - len(np.unique(A_label))
  if num_cont > 0:
    dup_list = []
    for ele in np.unique(A_label):
      if sum(A_label == ele) > 1:
        dup_list.append([np.where(A_label == ele)[0]])

    cont_ind = np.array(dup_list).reshape(2 * num_cont, order='F')
    free_ind = np.delete(np.arange(len(A_label)), cont_ind)

    cont_dim = np.prod(np.array(A.shape)[cont_ind[:num_cont]])
    free_dim = np.array(A.shape)[free_ind]

    B_label = np.delete(A_label, cont_ind)
    cont_label = np.unique(A_label[cont_ind])
    B = np.zeros(np.prod(free_dim))
    A = A.transpose(np.append(free_ind, cont_ind)).reshape(
        np.prod(free_dim), cont_dim, cont_dim)
    for ip in range(cont_dim):
      B = B + A[:, ip, ip]

    return B.reshape(free_dim), B_label, cont_label

  else:
    return A, A_label, []




 new_con_order.remove(k)





def ord_to_ncon(labels: List[List[int]], orders: np.ndarray):
  """
  Produces a `ncon` compatible index contraction order from the sequence of
  pairwise contractions.
  Args:
    labels: list of the tensor connections (in standard `ncon` format).
    orders: array of dim (2,N-1) specifying the set of N-1 pairwise
      tensor contractions.
  Returns:
    np.ndarray: the contraction order (in `ncon` format).
  """


# remove partial traces from connects and con_order
for ele in range(len(tensor_list)):
  num_cont = len(connect_list[ele]) - len(np.unique(connect_list[ele]))
  if num_cont > 0:
    tensor_list[ele], connect_list[ele], cont_ind = partial_trace(
        tensor_list[ele], connect_list[ele])
    con_order = np.delete(
        con_order,
        np.intersect1d(con_order, cont_ind, return_indices=True)[1])













# generate contraction order if necessary
flat_connect = np.concatenate(connect_list)
if con_order is None:
  con_order = np.unique(flat_connect[flat_connect > 0])
else:
  con_order = np.array(con_order)

# check inputs if enabled
if check_network:
  dims_list = [list(tensor.shape) for tensor in tensor_list]
  check_inputs(connect_list, flat_connect, dims_list, con_order)

# do all partial traces
for ele in range(len(tensor_list)):
  num_cont = len(connect_list[ele]) - len(np.unique(connect_list[ele]))
  if num_cont > 0:
    tensor_list[ele], connect_list[ele], cont_ind = partial_trace(
        tensor_list[ele], connect_list[ele])
    con_order = np.delete(
        con_order,
        np.intersect1d(con_order, cont_ind, return_indices=True)[1])

# do all binary contractions
while len(con_order) > 0:
  # identify tensors to be contracted
  cont_ind = con_order[0]
  locs = [
      ele for ele in range(len(connect_list))
      if sum(connect_list[ele] == cont_ind) > 0
  ]

  # do binary contraction
  cont_many, A_cont, B_cont = np.intersect1d(
      connect_list[locs[0]],
      connect_list[locs[1]],
      assume_unique=True,
      return_indices=True)
  if np.size(tensor_list[locs[0]]) < np.size(tensor_list[locs[1]]):
    ind_order = np.argsort(A_cont)
  else:
    ind_order = np.argsort(B_cont)

  tensor_list.append(
      np.tensordot(
          tensor_list[locs[0]],
          tensor_list[locs[1]],
          axes=(A_cont[ind_order], B_cont[ind_order])))
  connect_list.append(
      np.append(
          np.delete(connect_list[locs[0]], A_cont),
          np.delete(connect_list[locs[1]], B_cont)))

  # remove contracted tensors from list and update con_order
  del tensor_list[locs[1]]
  del tensor_list[locs[0]]
  del connect_list[locs[1]]
  del connect_list[locs[0]]
  con_order = np.delete(
      con_order,
      np.intersect1d(con_order, cont_many, return_indices=True)[1])

# do all outer products
while len(tensor_list) > 1:
  s1 = tensor_list[-2].shape
  s2 = tensor_list[-1].shape
  tensor_list[-2] = np.outer(tensor_list[-2].reshape(np.prod(s1)),
                             tensor_list[-1].reshape(np.prod(s2))).reshape(
                                 np.append(s1, s2))
  connect_list[-2] = np.append(connect_list[-2], connect_list[-1])
  del tensor_list[-1]
  del connect_list[-1]

# do final permutation
if len(connect_list[0]) > 0:
  return np.transpose(tensor_list[0], np.argsort(-connect_list[0]))
else:
  return tensor_list[0].item()


