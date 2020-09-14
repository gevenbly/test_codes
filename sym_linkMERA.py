"""
sym_linkMERA.py
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
from tensornetwork.block_sparse.charge import U1Charge
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse import BlockSparseTensor as BT
from tensornetwork.block_sparse import linalg as BLA
import tensornetwork as tn

# tensornetwork and block_sparse modules
from binaryMERA import binaryMERA
from eigh import eigh
from link_networks import LiftHam, LowerDensity

from helper_functs import orthog, trunct_eigh, expand_dims

tn.set_default_backend('symmetric')

# set simulation parameters
chi_b = 4
chi = 4
chi_p = 4
chi_m = 12
n_levels = 3
n_iterations = 2000
n_sites = 3 * (2**(n_levels + 1))

initialize_on = 1

# define quantum numbers
q_chi_b = U1Charge([-1, 0, 0, 1])
q_chi = U1Charge([-1, 0, 0, 1])
q_chi_p = U1Charge([-1, 0, 0, 1])

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

if initialize_on:
  # initialize tensors
  u = [0] * n_levels
  w = [0] * n_levels
  utemp = np.eye(chi_b**2, chi_b**2).reshape(chi_b, chi_b, chi_b, chi_b)
  u[0] = BT.fromdense([ind_chib1] * 2 + [ind_chib0] * 2, utemp)
  wtemp = orthog(np.random.rand(chi_b, chi_b, chi), pivot=2)
  w[0] = BT.fromdense([ind_chib1] * 2 + [ind_chi0], wtemp)
  for k in range(n_levels - 1):
    utemp = (np.eye(chi**2, chi_p**2)).reshape(chi, chi, chi_p, chi_p)
    u[k + 1] = BT.fromdense([ind_chi1] * 2 + [ind_chip0] * 2, utemp)
    wtemp = orthog(np.random.rand(chi_p, chi_p, chi), pivot=2)
    w[k + 1] = BT.fromdense([ind_chip1] * 2 + [ind_chi0], wtemp)

  vtemp = np.random.rand(chi, chi, chi, 1)
  vtemp = vtemp / LA.norm(vtemp)
  v = BT.fromdense([ind_chi1] * 3 + [ind_1], vtemp)

# warm-up sweep
ham = [0] * (n_levels + 1)
ham[0] = ham_init
for z in range(n_levels):
  ham[z + 1] = 2 * LiftHam(ham[z], u[z], w[z])

# diagonalize top level Hamiltonian, find GS within the charge=0 sector
ham_top = (ham[n_levels] +
           ham[n_levels].transpose([1, 2, 0, 4, 5, 3]) +
           ham[n_levels].transpose([2, 0, 1, 5, 3, 4]))
_, v = eigh(ham_top, link_charges=U1Charge([0]))


# lower the density matrix
rho = [0] * (n_levels + 1)
rho[n_levels] = tn.ncon([v.conj(), v], [[-1, -2, -3, 1], [-4, -5, -6, 1]])
for z in reversed(range(n_levels)):
  rho[z] = LowerDensity(u[z], w[z], rho[z + 1])

# compute energy and magnetization
energy_per_site = tn.ncon([ham_init, rho[0]],
                          [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).item() / 2
energy_exact = (-4 / np.sin(np.pi / n_sites)) / n_sites  # PBC (XX model)
energy_error = energy_per_site - energy_exact
bias0 = 2 * energy_per_site
eye_temp = (bias0 * np.eye(chi_b**3)).reshape([chi_b] * 6)
ham[0] = ham[0] - BT.fromdense([ind_chib1] * 3 + [ind_chib0] * 3, eye_temp)

print('Iteration: %d of %d, Energy: %f, Energy Error: %e\n'
      % (0, n_iterations, energy_per_site, energy_error))

# do iterations of single tensor optimizations
en_keep1 = [0] * n_iterations
for p in range(n_iterations):
  for z in range(n_levels):
    ham[z + 1] = 2 * LiftHam(ham[z], u[z], w[z])

  # diagonalize top level Hamiltonian, find GS within the charge=0 sector
  ham_top = (ham[n_levels] +
             ham[n_levels].transpose([1, 2, 0, 4, 5, 3]) +
             ham[n_levels].transpose([2, 0, 1, 5, 3, 4]))
  _, v = eigh(ham_top, link_charges=U1Charge([0]))

  # lower the density matrix
  rho[n_levels] = tn.ncon([v.conj(), v], [[-1, -2, -3, 1], [-4, -5, -6, 1]])
  for z in reversed(range(n_levels)):
    rho[z] = LowerDensity(u[z], w[z], rho[z + 1])

  # compute and display the energy
  energy_per_site = tn.ncon(
    [ham_init, rho[0]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).item() / 2
  en_keep1[p] = energy_per_site
  energy_error = energy_per_site - energy_exact
  print('Iteration: %d of %d, Energy: %f, Energy Error: %e\n'
        % (p, n_iterations, energy_per_site, energy_error))

  bias0 = tn.ncon([ham[0], rho[0]],
                  [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).item()
  eye_temp = (bias0 * np.eye(chi_b**3)).reshape([chi_b] * 6)
  ham[0] = ham[0] - BT.fromdense([ind_chib1] * 3 + [ind_chib0] * 3, eye_temp)




