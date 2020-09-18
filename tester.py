# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:44:22 2020

@author: gevenbly3
"""



# general python modules
import numpy as np
from numpy import linalg as LA

# tensornetwork and block_sparse modules
from tensornetwork.block_sparse.charge import U1Charge
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse import BlockSparseTensor as BT
import tensornetwork as tn

tn.set_default_backend('symmetric')

# set simulation parameters
chi_b = 4
chi = 4
chi_p = 4
chi_m = 12
n_levels = 3
n_iterations = 2000
n_sites = 3 * (2**(n_levels + 1))

left_on = 1
right_on = 1
center_on = 1
initialize_on = 1

# define quantum numbers
q_chi_b = U1Charge([-1, 0, 0, 1])
q_chi_p = U1Charge([-1, 0, 0, 1])
q_chi = U1Charge([-1, 0, 0, 1])

# create 2 versions of each index (incoming/outgoing flows)
# --- Martin: is there a way of avoiding defining both flows separately?
ind_chib0 = Index(charges=q_chi_b, flow=True)
ind_chib1 = Index(charges=q_chi_b, flow=False)
ind_chip0 = Index(charges=q_chi_p, flow=True)
ind_chip1 = Index(charges=q_chi_p, flow=False)
ind_chi0 = Index(charges=q_chi, flow=True)
ind_chi1 = Index(charges=q_chi, flow=False)
ind_1 = Index(charges=U1Charge([0]), flow=False)
# -----------------

# define hamiltonian and do 2-to-1 blocking
chi_b = 4
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0, -1]])
ham_s = np.real(np.kron(sX, sX) + np.kron(sY, sY))
# ham_s = (-np.kron(sX, sX) + 0.5 * np.kron(np.eye(2), sZ) +
#          0.5 * np.kron(np.eye(2), sZ))
ham_temp = (0.5 * np.kron(np.eye(8), np.kron(ham_s, np.eye(2))) +
            np.kron(np.eye(4), np.kron(ham_s, np.eye(4))) +
            0.5 * np.kron(np.eye(2), np.kron(ham_s, np.eye(8))))
ham_init = BT.fromdense([ind_chib1] * 3 + [ind_chib0] * 3,
                        ham_temp.reshape([chi_b] * 6))
bias_shift = max(LA.eigvalsh(ham_temp)) - min(LA.eigvalsh(ham_temp))
ham = [0] * (n_levels + 1)
ham[0] = ham_init.copy()

if initialize_on:
  # initialize tensors
  u = [0] * n_levels
  w = [0] * n_levels
  utemp = np.eye(chi_b**2, chi_b**2).reshape(chi_b, chi_b, chi_b, chi_b)
  u[0] = BT.fromdense([ind_chib1] * 2 + [ind_chib0] * 2, utemp)
  w[0] = BT.randn([ind_chib1] * 2 + [ind_chib0], dtype='float64')
  for k in range(n_levels - 1):
    utemp = (np.eye(chi**2, chi_p**2)).reshape(chi, chi, chi_p, chi_p)
    u[k + 1] = BT.fromdense([ind_chi1] * 2 + [ind_chip0] * 2, utemp)
    w[k + 1] = BT.randn([ind_chip1] * 2 + [ind_chi0], dtype='float64')


#######################################
def LiftHam(ham, u, w):

  tensors = [ham,u,u,u.conj(),u.conj(),w,w,w,w.conj(),w.conj(),w.conj()] 
  
  # Network 1 - leading cost: (chi^4)*(chi_p^5) 
  connects1 = [[5,6,7,2,3,4],[1,2,9,10],[3,4,11,12],[1,5,13,14],[6,7,15,16],
              [10,11,-5],[8,9,-4],[12,17,-6],[14,15,-2],[8,13,-1],[16,17,-3]] 
  con_order1 = [10,6,7,3,4,2,11,17,8,5,1,15,14,16,12,9,13]
  
  # Network 2 - leading cost: (chi^4)*(chi_p^5) 
  connects2 = [[4,5,6,1,2,3],[1,2,8,9],[3,17,10,11],[4,5,12,13],[6,17,14,15],
              [9,10,-5],[7,8,-4],[11,16,-6],[13,14,-2],[7,12,-1],[15,16,-3]] 
  con_order2 = [14,4,5,1,2,16,6,13,7,3,17,9,10,15,11,12,8]
  
  return (tn.ncon(tensors,connects1,con_order1) +
          tn.ncon(tensors,connects2,con_order2))
#######################################



# This code works
ham[1] = 2 * LiftHam(ham[0], u[0], w[0])
ham[2] = 2 * LiftHam(ham[1], u[1], w[1])


# This code does not work
ham[1] = 2 * LiftHam(ham[0], u[0], w[0])
ham_temp = ham[1] - BT.fromdense(ham[1].sparse_shape, np.random.rand(4,4,4,4,4,4))
ham[2] = 2 * LiftHam(ham[1], u[1], w[1])