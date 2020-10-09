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
import tensornetwork as tn
from tensornetwork.block_sparse.charge import U1Charge
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse import BlockSparseTensor as BT
from tensornetwork.block_sparse import linalg as BLA
from tensornetwork.block_sparse import initialization as BI
from tensornetwork.block_sparse.caching import set_caching_status, clear_cache


# tensornetwork and block_sparse modules
from eigh import eigh
from helper_functs import orthog, trunct_eigh, expand_dims, orthog_sym, eye_sym
from sym_link_networks import (LiftHam, LowerDensity, RightLink, LeftLink,
                           TopLink,CenterLink)
# from link_networks import CenterLink
from binaryMERA import binaryMERA

tn.set_default_backend('symmetric')
set_caching_status(True)
clear_cache()

left_on = 1
right_on = 1
center_on = 1
initialize_on = 0

prelim_sweeps = 20
link_sweep_gap = 4

# set simulation parameters
chi = 12
n_levels = 3
n_iterations = 100
n_sites = 3 * (2**(n_levels + 1))

# ------------------------------------------------------------
# initialization

# define quantum numbers
q_chi_b = U1Charge([-2, 0, 0, 2])
ind_chib0 = Index(charges=q_chi_b, flow=True)
ind_chib1 = Index(charges=q_chi_b, flow=False)

# define hamiltonian and do 2-to-1 blocking
chi_b = 4
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0, -1]])
ham_s = np.real(np.kron(sX, sX) + np.kron(sY, sY))
ham_temp = (0.5 * np.kron(np.eye(8), np.kron(ham_s, np.eye(2))) +
            np.kron(np.eye(4), np.kron(ham_s, np.eye(4))) +
            0.5 * np.kron(np.eye(2), np.kron(ham_s, np.eye(8))))
bias_shift = max(LA.eigvalsh(ham_temp)) - min(LA.eigvalsh(ham_temp))
ham_init = BT.fromdense([ind_chib1] * 3 + [ind_chib0] * 3,
                        ham_temp.reshape([chi_b] * 6))

if initialize_on:
  # initialize tensors
  u = [0] * n_levels
  w = [0] * n_levels
  utemp = np.eye(chi_b**2, chi_b**2).reshape(chi_b, chi_b, chi_b, chi_b)
  u[0] = BT.fromdense([ind_chib1] * 2 + [ind_chib0] * 2, utemp)
  w[0] = orthog_sym(BT.randn([ind_chib1] * 2 + [ind_chib0],
                             dtype='float64'), pivot=2)
  for k in range(n_levels - 1):
    utemp = (np.eye(chi_b**2, chi_b**2)).reshape(chi_b, chi_b, chi_b, chi_b)
    u[k + 1] = BT.fromdense([ind_chib1] * 2 + [ind_chib0] * 2, utemp)
    w[k + 1] = orthog_sym(BT.randn([ind_chib1] * 2 + [ind_chib0],
                                   dtype='float64'), pivot=2)

  vtemp = np.random.rand(chi_b, chi_b, chi_b, 1)
  vtemp = vtemp / LA.norm(vtemp)
  v = BT.fromdense([ind_chib1] * 3 + [Index(charges=U1Charge([0]),
                                            flow=False)], vtemp)
else:
  # pass
  u, w, v, rho, ham = np.load('tempdata.npy', allow_pickle=True)
  # u, w, v, rho, ham = np.load('XXData_temp4.npy', allow_pickle=True)
# np.save('tempdata.npy', (u, w, v, rho, ham))
# ------------------------------------------------------------

# ------------------------------------------------------------
# warm-up sweep
ham = [0] * (n_levels + 1)
ham[0] = ham_init.copy()
for z in range(n_levels):
  ham[z + 1] = 2 * LiftHam(ham[z], u[z], w[z])

# diagonalize top level Hamiltonian, find GS within the charge=0 sector
ham_top = (ham[n_levels] +
           ham[n_levels].transpose([1, 2, 0, 4, 5, 3]) +
           ham[n_levels].transpose([2, 0, 1, 5, 3, 4]))
_, v = eigh(ham_top, link_charges=U1Charge([0]))

# lower the density matrix, compute spectrum of 1-site density
rho = [0] * (n_levels + 1)
rho[n_levels] = tn.ncon([v.conj(), v], [[-1, -2, -3, 1], [-4, -5, -6, 1]])
spect_chi = [0] * (n_levels + 1)
sp_temp, _ = eigh(tn.ncon([rho[n_levels]],
                          [[-1, 1, 2, -2, 1, 2]]), full_sort=True)
spect_chi[n_levels] = sp_temp.todense()
for z in reversed(range(n_levels)):
  rho[z] = LowerDensity(u[z], w[z], rho[z + 1])
  sp_temp, _ = eigh(tn.ncon([rho[z]], [[-1, 1, 2, -2, 1, 2]]), full_sort=True)
  spect_chi[z] = sp_temp.todense()

# compute energy and magnetization
energy_per_site = tn.ncon([ham_init, rho[0]],
                          [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).item() / 2
energy_exact = (-4 / np.sin(np.pi / n_sites)) / n_sites  # PBC (XX model)
energy_error = energy_per_site - energy_exact

# shift spectrum to zero the energy
bias0 = 2 * energy_per_site
eye_temp = (bias0 * np.eye(chi_b**3)).reshape([chi_b] * 6)
ham[0] = ham[0] - BT.fromdense([ind_chib1] * 3 + [ind_chib0] * 3, eye_temp)

print('Iteration: %d of %d, Energy: %f, Energy Error: %e\n'
      % (0, n_iterations, energy_per_site, energy_error))
# ------------------------------------------------------------

link_type = 0
en_keep = [0] * n_iterations
for p in range(n_iterations):
  en_bias = tn.ncon([ham[0], rho[0]], [[1, 2, 3, 4, 5, 6],
                                       [1, 2, 3, 4, 5, 6]]).item()
  chi_temp = ham[0].shape[0]
  eye_temp = np.eye(chi_temp**3).reshape([chi_temp] * 6)
  ham[0] = ham[0] - BT.fromdense(ham[0].sparse_shape, en_bias * eye_temp)

  for z in range(n_levels):
    chi_temp = ham[z].shape[0]
    eye_temp = 4 * bias_shift * np.eye(chi_temp**3).reshape([chi_temp] * 6)
    ham_temp = ham[z] - BT.fromdense(ham[z].sparse_shape, eye_temp)

    thres_z0 = min(spect_chi[z])
    thres_z1 = min(spect_chi[z + 1])

    if p > prelim_sweeps and np.mod(p, link_sweep_gap) == 0:
      link_sweep = True
    else:
      link_sweep = False

    if link_sweep:
      # print(u[z].shape)

      if left_on and link_type == 0:
        u[z], w[z] = LeftLink(ham_temp, u[z], w[z], rho[z + 1], thres_z0)

      if right_on and link_type == 1:
        u[z], w[z] = RightLink(ham_temp, u[z], w[z], rho[z + 1], thres_z0)

      if center_on and link_type == 2:
        if z < (n_levels - 1):
          w[z], u[z + 1] = CenterLink(ham_temp, u[z], w[z], u[z + 1], w[z + 1],
                                      rho[z + 2], chi, 0)
        else:
          phitemp = TopLink(ham_temp, u[z], w[z], v, chi)
          rhotemp = tn.ncon([phitemp.conj(), phitemp],
                            [[-1, -2, 1, 2, 3], [-3, -4, 1, 2, 3]])
          _, w[z] = eigh(rhotemp, which='LM', max_kept=chi, full_sort=True)
    else:
      if p > 10:
        # optimise disentanglers
        tensor_list = [u[z], w[z], ham_temp, rho[z + 1]]
        u_env = (binaryMERA(tensor_list, which_network=1, which_env=1) +
                 binaryMERA(tensor_list, which_network=1, which_env=2) +
                 binaryMERA(tensor_list, which_network=2, which_env=1) +
                 binaryMERA(tensor_list, which_network=2, which_env=2))
        u[z] = orthog_sym(-u_env.conj(), pivot=2)

      # optimise isometries
      tensor_list = [u[z], w[z], ham_temp, rho[z + 1]]
      w_env = (binaryMERA(tensor_list, which_network=1, which_env=3) +
               binaryMERA(tensor_list, which_network=1, which_env=4) +
               binaryMERA(tensor_list, which_network=1, which_env=5) +
               binaryMERA(tensor_list, which_network=2, which_env=3) +
               binaryMERA(tensor_list, which_network=2, which_env=4) +
               binaryMERA(tensor_list, which_network=2, which_env=5))
      w[z] = orthog_sym(-w_env.conj(), pivot=2)

    ham[z + 1] = 2 * LiftHam(ham[z], u[z], w[z])

  # diagonalize top level Hamiltonian, find GS within the charge=0 sector
  ham_top = (ham[n_levels] +
             ham[n_levels].transpose([1, 2, 0, 4, 5, 3]) +
             ham[n_levels].transpose([2, 0, 1, 5, 3, 4]))
  _, v = eigh(ham_top, link_charges=U1Charge([0]), which='SA')

  # lower the density matrix
  rho[n_levels] = tn.ncon([v.conj(), v], [[-1, -2, -3, 1], [-4, -5, -6, 1]])
  spect_chi = [0] * (n_levels + 1)
  sp_temp, _ = eigh(tn.ncon([rho[n_levels]], [[-1, 1, 2, -2, 1, 2]]),
                    full_sort=True)
  spect_chi[n_levels] = sp_temp.todense()
  for z in reversed(range(n_levels)):
    rho[z] = LowerDensity(u[z], w[z], rho[z + 1])
    sp_temp, _ = eigh(tn.ncon([rho[z]], [[-1, 1, 2, -2, 1, 2]]), full_sort=True)
    spect_chi[z] = sp_temp.todense()

  # compute and display the energy
  energy_per_site = 0.5 * tn.ncon([ham_init, rho[0]],
                                  [[1, 2, 3, 4, 5, 6],
                                   [1, 2, 3, 4, 5, 6]]).item()
  en_keep[p] = energy_per_site
  energy_error = energy_per_site - energy_exact
  print('Iteration: %d of %d, Energy: %f, Energy Error: %e\n'
        % (p, n_iterations, energy_per_site, energy_error))

  bias0 = tn.ncon([ham[0], rho[0]],
                  [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).item()
  eye_temp = (bias0 * np.eye(chi_b**3)).reshape([chi_b] * 6)
  ham[0] = ham[0] - BT.fromdense([ind_chib1] * 3 + [ind_chib0] * 3, eye_temp)

  link_type += 1
  if link_type > 2:
    link_type = 0
