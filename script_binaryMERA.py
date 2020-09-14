"""
script_binaryMERA.py
---------------------------------------------------------------------
Script file demonstrating how the auto-generated code from the "TensorTrace"
software can be implemented in a tensor network algorithm (in this case
an optimization of a binary MERA for the ground state of the 1D transverse
field quantum Ising model on a finite lattice). This script calls the
"binaryMERA.py" function file as automatically generated from the example
TensorTrace project "binaryMERA.ttp" included with the distribution.

by Glen Evenbly (www.glenevenbly.com) - last modified 06/2020
"""

from scipy.sparse.linalg import eigsh
from numpy import linalg as LA
import numpy as np
from binaryMERA import binaryMERA
from link_networks import LiftHam, LowerDensity
import tensornetwork as tn
from helper_functs import orthog, expand_dims

# set simulation parameters
chi = 6
chi_p = 4
n_levels = 3
n_iterations = 1000
n_sites = 3 * (2**(n_levels + 1))

initialize_on = 1

# define hamiltonian and do 2-to-1 blocking
# chi_b = 4
# sX = np.array([[0, 1], [1, 0]])
# sZ = np.array([[1, 0], [0, -1]])
# ham_s = (-np.kron(sX, sX) + 0.5 * np.kron(np.eye(2), sZ) +
#          0.5 * np.kron(np.eye(2), sZ))
# ham_init = (0.5 * np.kron(np.eye(8), np.kron(ham_s, np.eye(2))) +
#             np.kron(np.eye(4), np.kron(ham_s, np.eye(4))) +
#             0.5 * np.kron(np.eye(2), np.kron(ham_s, np.eye(8))))
chi_b = 4
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0, -1]])
ham_s = np.real(np.kron(sX, sX) + np.kron(sY, sY))
ham_init = (0.5 * np.kron(np.eye(8), np.kron(ham_s, np.eye(2))) +
            np.kron(np.eye(4), np.kron(ham_s, np.eye(4))) +
            0.5 * np.kron(np.eye(2), np.kron(ham_s, np.eye(8))))
# bias = max(LA.eigvalsh(ham_init))
bias_shift = max(LA.eigvalsh(ham_init)) - min(LA.eigvalsh(ham_init))

if initialize_on:
  # initialize tensors
  u = [0] * n_levels
  w = [0] * n_levels
  u[0] = np.eye(chi_b**2, chi_b**2).reshape(chi_b, chi_b, chi_b, chi_b)
  w[0] = orthog(np.random.rand(chi_b, chi_b, chi), pivot=2)
  for k in range(n_levels - 1):
    u[k + 1] = (np.eye(chi**2, chi_p**2)).reshape(chi, chi, chi_p, chi_p)
    w[k + 1] = orthog(np.random.rand(chi_p, chi_p, chi), pivot=2)

  v = np.random.rand(chi, chi, chi)
  v = v / LA.norm(v)

else:
  # expand tensor dimensions
  u[0] = expand_dims(u[0], [chi_b, chi_b, chi_b, chi_b])
  w[0] = expand_dims(w[0], [chi_b, chi_b, chi])
  for k in range(n_levels - 1):
    u[k + 1] = expand_dims(u[k + 1], [chi, chi, chi_p, chi_p])
    w[k + 1] = expand_dims(w[k + 1], [chi_p, chi_p, chi])

# warm-up sweep
ham = [0] * (n_levels + 1)
ham[0] = ham_init.reshape([chi_b] * 6)
for z in range(n_levels):
  ham[z + 1] = 2 * LiftHam(ham[z], u[z], w[z])

# diagonalize top-level Hamiltonian
ham_top = (ham[n_levels] +
           ham[n_levels].transpose(1, 2, 0, 4, 5, 3) +
           ham[n_levels].transpose(2, 0, 1, 5, 3, 4)
           ).reshape(chi**3, chi**3)
dtemp, vtemp = eigsh(0.5 * (ham_top + np.conj(ham_top.T)), k=1, which='SA')
vtemp = vtemp / LA.norm(vtemp)
v = vtemp.reshape([chi] * 3)

# lower the density matrix
rho = [0] * (n_levels + 1)
rho[n_levels] = (vtemp @ np.conj(vtemp.T)).reshape([chi] * 6)
for z in reversed(range(n_levels)):
  rho[z] = LowerDensity(u[z], w[z], rho[z + 1])

# compute energy and magnetization
energy_per_site = np.trace(rho[0].reshape(chi_b**3, chi_b**3) @ ham_init) / 2
expectX = np.trace(rho[0].reshape(chi_b**3, chi_b**3) @
                   np.kron(np.eye(32), sX))
# energy_exact = (-2 / np.sin(np.pi / (2 * n_sites))) / n_sites  # PBC (Ising)
energy_exact = (-4 / np.sin(np.pi / n_sites)) / n_sites  # PBC (XX model)
energy_error = energy_per_site - energy_exact
bias0 = 2 * energy_per_site
ham[0] = ham[z] - (bias0 * np.eye(chi_b**3)).reshape([chi_b] * 6)

print('Iteration: %d of %d, Energy: %f, Energy Error: %e, XMag: %e\n'
      % (0, n_iterations, energy_per_site, energy_error, expectX))

# do iterations of single tensor optimizations
en_keep0 = [0] * n_iterations
for p in range(n_iterations):

  # sweep over all levels
  for z in range(n_levels):
    bias = tn.ncon([ham[z], rho[z]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    # print('bias: %f \n' % (bias))
    chi_temp = ham[z].shape[0]
    ham_temp = ham[z] - (((bias + bias_shift) *
                          np.eye(chi_temp**3)).reshape([chi_temp] * 6))

    # optimise disentanglers
    tensor_list = [u[z], w[z], ham_temp, rho[z + 1]]
    uEnv = (binaryMERA(tensor_list, which_network=1, which_env=1) +
            binaryMERA(tensor_list, which_network=1, which_env=2) +
            binaryMERA(tensor_list, which_network=2, which_env=1) +
            binaryMERA(tensor_list, which_network=2, which_env=2))
    uSize = uEnv.shape
    uEnv = uEnv.reshape(uSize[0] * uSize[1], uSize[2] * uSize[3])
    utemp, stemp, vtemph = LA.svd(uEnv, full_matrices=False)
    u[z] = (-utemp @ vtemph).reshape(uSize)

    # optimise isometries
    tensor_list = [u[z], w[z], ham_temp, rho[z + 1]]
    wEnv = (binaryMERA(tensor_list, which_network=1, which_env=3) +
            binaryMERA(tensor_list, which_network=1, which_env=4) +
            binaryMERA(tensor_list, which_network=1, which_env=5) +
            binaryMERA(tensor_list, which_network=2, which_env=3) +
            binaryMERA(tensor_list, which_network=2, which_env=4) +
            binaryMERA(tensor_list, which_network=2, which_env=5))
    wSize = wEnv.shape
    wEnv = wEnv.reshape(wSize[0] * wSize[1], wSize[2])
    utemp, stemp, vtemph = LA.svd(wEnv, full_matrices=False)
    w[z] = (-utemp @ vtemph).reshape(wSize)

    # lift Hamiltonian
    tensor_list = [u[z], w[z], ham[z], rho[z + 1]]
    ham[z + 1] = 2 * (binaryMERA(tensor_list, which_network=1, which_env=12) +
                      binaryMERA(tensor_list, which_network=2, which_env=12))

  # diagonalize top-level Hamiltonian
  ham_top = (ham[n_levels] +
             ham[n_levels].transpose(1, 2, 0, 4, 5, 3) +
             ham[n_levels].transpose(2, 0, 1, 5, 3, 4)
             ).reshape(chi**3, chi**3)
  dtemp, vtemp = eigsh(0.5 * (ham_top + np.conj(ham_top.T)), k=1, which='SA')
  vtemp = vtemp / LA.norm(vtemp)
  top_shape = [chi] * 6
  v = vtemp.reshape(chi, chi, chi)

  # lower the density matrix
  rho[n_levels] = (vtemp @ np.conj(vtemp.T)).reshape([chi] * 6)
  for z in reversed(range(n_levels)):
    tensor_list = [u[z], w[z], ham[z], rho[z + 1]]
    rho[z] = 0.5 * (binaryMERA(tensor_list, which_network=1, which_env=6) +
                    binaryMERA(tensor_list, which_network=2, which_env=6))

  # compute energy and magnetization
  energy_per_site = np.trace(rho[0].reshape(chi_b**3, chi_b**3) @ ham_init) / 2
  en_keep0[p] = energy_per_site
  expectX = np.trace(rho[0].reshape(chi_b**3, chi_b**3) @
                     np.kron(np.eye(32), sX))
  energy_error = energy_per_site - energy_exact
  print('Iteration: %d of %d, Energy: %f, Energy Error: %e, XMag: %e\n'
        % (p, n_iterations, energy_per_site, energy_error, expectX))

  bias0 = tn.ncon([ham[z], rho[z]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
  ham[0] = ham[z] - (bias0 * np.eye(chi_b**3)).reshape([chi_b] * 6)


# np.save('XXData0_bin.npy', (en_keep0))
# en_keep0 = np.load('XXData0_bin.npy', allow_pickle=True)
