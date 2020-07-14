"""
sym_MERA.py
---------------------------------------------------------------------
Script file for running a Z2 symmetric MERA using the block_sparse library
from tensornetwork. Uses the auto-generated "binaryMERA.py" code from the
"TensorTrace" software, which contains the relevant network contractions.

by Glen Evenbly (www.glenevenbly.com) - last modified 06/2020
"""

# general python modules
import numpy as np
from numpy import linalg as LA

# tensornetwork and block_sparse modules
from tensornetwork.block_sparse.charge import Z2Charge
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse import BlockSparseTensor as BT
from tensornetwork.block_sparse import linalg as BLA
import tensornetwork as tn

# tensornetwork and block_sparse modules
from binaryMERA import binaryMERA
from eigh import eigh

tn.set_default_backend('symmetric')

# set simulation parameters
chib = 4
chim = 4
chi = 6
n_levels = 3
n_iterations = 2000
n_sites = 3 * (2**(n_levels + 1))

# define quantum numbers
q_chib = Z2Charge([0, 1, 1, 0])
q_chim = Z2Charge([0] * (chim // 2) + [1] * (chim // 2))
q_chi = Z2Charge([0] * (chi // 2) + [1] * (chi // 2))

# create 2 versions of each index (incoming/outgoing flows)
ind_chib0 = Index(charges=q_chib, flow=True)
ind_chib1 = Index(charges=q_chib, flow=False)
ind_chim0 = Index(charges=q_chim, flow=True)
ind_chim1 = Index(charges=q_chim, flow=False)
ind_chi0 = Index(charges=q_chi, flow=True)
ind_chi1 = Index(charges=q_chi, flow=False)

# define hamiltonian and do 2-to-1 blocking
en_exact = (-2 / np.sin(np.pi / (2 * n_sites))) / n_sites
sX = np.array([[0, 1], [1, 0]])
sZ = np.array([[1, 0], [0, -1]])
ham_s = (-np.kron(sX, sX) + 0.5 * np.kron(np.eye(2), sZ) +
         0.5 * np.kron(np.eye(2), sZ))
ham_init = (0.5 * np.kron(np.eye(8), np.kron(ham_s, np.eye(2))) +
            np.kron(np.eye(4), np.kron(ham_s, np.eye(4))) +
            0.5 * np.kron(np.eye(2), np.kron(ham_s, np.eye(8))))
bias = max(LA.eigvalsh(ham_init))
ham_z0 = BT.fromdense([ind_chib1] * 3 + [ind_chib0] * 3,
                      ham_init.reshape([chib] * 6))

# initialize tensors
uDis = [0] * n_levels
wIso = [0] * n_levels
ham = [0] * (n_levels + 1)
rho = [0] * (n_levels + 1)

eye_mat = np.eye(chib**2, chib**2).reshape(chib, chib, chib, chib)
uDis[0] = BT.fromdense([ind_chib1, ind_chib1, ind_chib0, ind_chib0], eye_mat)

w_temp = BT.randn([ind_chib1, ind_chib1, ind_chi0], dtype=np.float64)
ut, st, vt = BLA.svd(w_temp.reshape([chib**2, chi]), full_matrices=False)
wIso[0] = (ut @ vt).reshape([chib, chib, chi])

ham_temp = (ham_init - bias * np.eye(chib**3)).reshape([chib] * 6)
ham[0] = BT.fromdense([ind_chib1] * 3 + [ind_chib0] * 3, ham_temp)

rho[0] = BT.randn([ind_chib0] * 3 + [ind_chib1] * 3, dtype=np.float64)

for z in range(1, n_levels):
  eye_mat = np.eye(chim**2, chi**2).reshape(chi, chi, chim, chim)
  uDis[z] = BT.fromdense([ind_chi1, ind_chi1, ind_chim0, ind_chim0], eye_mat)

  w_temp = BT.randn([ind_chim1, ind_chim1, ind_chi0], dtype=np.float64)
  ut, st, vt = BLA.svd(w_temp.reshape([chim**2, chi]), full_matrices=False)
  wIso[z] = (ut @ vt).reshape([chim, chim, chi])

  ham[z] = BT.randn([ind_chi1] * 3 + [ind_chi0] * 3, dtype=np.float64)

  rho[z] = BT.randn([ind_chi0] * 3 + [ind_chi1] * 3, dtype=np.float64)

ham[n_levels] = BT.randn([ind_chi1] * 3 + [ind_chi0] * 3, dtype=np.float64)
rho[n_levels] = BT.randn([ind_chi0] * 3 + [ind_chi1] * 3, dtype=np.float64)

# do iterations of single tensor optimizations
for p in range(n_iterations):

  # sweep over all levels
  for z in range(n_levels):

    if p > 10:
      # optimise disentanglers
      tensor_list = [uDis[z], wIso[z], ham[z], rho[z + 1]]
      uEnv = (binaryMERA(tensor_list, which_network=1, which_env=1) +
              binaryMERA(tensor_list, which_network=1, which_env=2) +
              binaryMERA(tensor_list, which_network=2, which_env=1) +
              binaryMERA(tensor_list, which_network=2, which_env=2))
      uSize = uEnv.shape
      uEnv = uEnv.reshape([uSize[0] * uSize[1], uSize[2] * uSize[3]])
      utemp, stemp, vtemph = BLA.svd(uEnv.conj(), full_matrices=False)
      uDis[z] = (-1 * utemp @ vtemph).reshape(uSize)

    # optimise isometries
    tensor_list = [uDis[z], wIso[z], ham[z], rho[z + 1]]
    wEnv = (binaryMERA(tensor_list, which_network=1, which_env=3) +
            binaryMERA(tensor_list, which_network=1, which_env=4) +
            binaryMERA(tensor_list, which_network=1, which_env=5) +
            binaryMERA(tensor_list, which_network=2, which_env=3) +
            binaryMERA(tensor_list, which_network=2, which_env=4) +
            binaryMERA(tensor_list, which_network=2, which_env=5))
    wSize = wEnv.shape
    wEnv = wEnv.reshape([wSize[0] * wSize[1], wSize[2]])
    utemp, stemp, vtemph = BLA.svd(wEnv.conj(), full_matrices=False)
    wIso[z] = (-1 * utemp @ vtemph).reshape(wSize)

    # lift Hamiltonian
    tensor_list = [uDis[z], wIso[z], ham[z], rho[z + 1]]
    ham[z + 1] = (binaryMERA(tensor_list, which_network=1, which_env=12) +
                  binaryMERA(tensor_list, which_network=2, which_env=12))

  # diagonalize top level Hamiltonian
  ham_top = (ham[n_levels] +
             ham[n_levels].transpose([1, 2, 0, 4, 5, 3]) +
             ham[n_levels].transpose([2, 0, 1, 5, 3, 4]))
  E, V = eigh(ham_top, link_charges=Z2Charge([0]))
  rho[n_levels] = tn.ncon([V.conj(), V], [[-1, -2, -3, 1], [-4, -5, -6, 1]])

  # lower the density matrix
  for z in reversed(range(n_levels)):
    tensor_list = [uDis[z], wIso[z], ham[z], rho[z + 1]]
    rho[z] = 0.5 * (binaryMERA(tensor_list, which_network=1, which_env=6) +
                    binaryMERA(tensor_list, which_network=2, which_env=6))

  # compute and display the energy
  en_persite = 0.5 * tn.ncon([ham_z0, rho[0]],
                             [[1, 2, 3, 4, 5, 6],
                              [1, 2, 3, 4, 5, 6]]).todense().item()
  en_error = en_persite - en_exact
  print('Iteration: %d of %d, Energy: %f, Energy Error: %e\n'
        % (p, n_iterations, en_persite, en_error))
