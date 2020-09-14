"""
script_linkMERA.py
---------------------------------------------------------------------
"""

from scipy.sparse.linalg import eigsh
from numpy import linalg as LA
import numpy as np
from helper_functs import orthog, trunct_eigh, expand_dims
import matplotlib.pyplot as plt

from link_networks import (LiftHam, LowerDensity, RightLink, LeftLink,
                           CenterLink, TopLink)
import tensornetwork as tn

# set simulation parameters
chi = 6
chi_p = 4
chi_m = 12
n_levels = 3
n_iterations = 1000
n_sites = 3 * (2**(n_levels + 1))

left_on = 1
right_on = 1
center_on = 1
initialize_on = 1

# define hamiltonian and do 2-to-1 blocking
chi_b = 4
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0, -1]])
ham_s = np.real(np.kron(sX, sX) + np.kron(sY, sY))
ham_init = (0.5 * np.kron(np.eye(8), np.kron(ham_s, np.eye(2))) +
            np.kron(np.eye(4), np.kron(ham_s, np.eye(4))) +
            0.5 * np.kron(np.eye(2), np.kron(ham_s, np.eye(8))))
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
# ham[0] = (ham_init - bias * np.eye(chi_b**3)).reshape([chi_b] * 6)
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
v = vtemp.reshape(chi, chi, chi)

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

# do optimization iterations
en_keep = [0] * n_iterations
for p in range(n_iterations):
  for z in range(n_levels):
    bias = tn.ncon([ham[z], rho[z]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    # print('bias: %f \n' % (bias))
    chi_temp = ham[z].shape[0]
    ham_temp = ham[z] - (((bias + bias_shift) *
                          np.eye(chi_temp**3)).reshape([chi_temp] * 6))

    if right_on:
      # RightLink
      gam1, gam2 = RightLink(ham_temp, u[z], w[z], rho[z + 1], chi_m)

      if (u[z].shape[3] < u[z].shape[1]):
        y = np.eye(u[z].shape[1], u[z].shape[3])
        upr = tn.ncon([u[z], y], [[-1, -2, -3, 1], [-4, 1]])
        wpr = tn.ncon([w[z], y], [[1, -2, -3], [-1, 1]])
        gam2pr = tn.ncon([gam2, y], [[1, -2, -3], [-1, 1]])
      else:
        upr = u[z]
        wpr = w[z]
        gam2pr = gam2

      for g in range(10):
        upr = orthog(tn.ncon([gam1, wpr], [[-1, -2, 1, -3, 2], [-4, 1, 2]]),
                     pivot=2)
        wpr = orthog(tn.ncon([gam1, upr], [[1, 2, -2, 3, -3], [1, 2, 3, -1]]) +
                     gam2pr, pivot=2)

      if (u[z].shape[3] < u[z].shape[1]):
        rhotemp = tn.ncon([wpr, wpr.conj(), rho[z + 1]],
                          [[-2, 5, 4], [-1, 5, 3], [3, 1, 2, 4, 1, 2]])
        dtemp, y = trunct_eigh(rhotemp, chi_p)
        u[z] = tn.ncon([upr, y], [[-1, -2, -3, 1], [1, -4]])
        w[z] = tn.ncon([wpr, y], [[1, -2, -3], [1, -1]])
      else:
        u[z] = upr
        w[z] = wpr

    if left_on:
      # LeftLink
      gam1, gam2 = LeftLink(ham_temp, u[z], w[z], rho[z + 1], chi_m)

      if (u[z].shape[2] < u[z].shape[0]):
        y = np.eye(u[z].shape[0], u[z].shape[2])
        upr = tn.ncon([u[z], y], [[-1, -2, 1, -4], [-3, 1]])
        wpr = tn.ncon([w[z], y], [[-1, 1, -3], [-2, 1]])
        gam2pr = tn.ncon([gam2, y], [[-1, 1, -3], [-2, 1]])
      else:
        upr = u[z]
        wpr = w[z]
        gam2pr = gam2

      for g in range(10):
        upr = orthog(tn.ncon([gam1, wpr], [[1, -1, -2, 2, -4], [1, -3, 2]]), 2)
        wpr = orthog(tn.ncon([gam1, upr], [[-1, 1, 2, -3, 3], [1, 2, -2, 3]]) +
                     gam2pr, 2)

      if (u[z].shape[2] < u[z].shape[0]):
        rhotemp = tn.ncon([wpr, wpr.conj(), rho[z + 1]],
                          [[5, -2, 4], [5, -1, 3], [3, 1, 2, 4, 1, 2]])
        dtemp, y = trunct_eigh(rhotemp, chi_p)
        u[z] = tn.ncon([upr, y], [[-1, -2, 1, -4], [1, -3]])
        w[z] = tn.ncon([wpr, y], [[-1, 1, -3], [1, -2]])
      else:
        u[z] = upr
        w[z] = wpr

    if center_on:
      # CenterLink
      if z < (n_levels - 1):
        w[z], u[z + 1] = CenterLink(ham_temp, u[z], w[z], u[z + 1], w[z + 1],
                                    rho[z + 2], chi, chi_m)
      else:
        w[z] = TopLink(ham_temp, u[z], w[z], v, chi)

    ham[z + 1] = 2 * LiftHam(ham[z], u[z], w[z])

  # diagonalize top-level Hamiltonian
  ham_top = (ham[n_levels] +
             ham[n_levels].transpose(1, 2, 0, 4, 5, 3) +
             ham[n_levels].transpose(2, 0, 1, 5, 3, 4)
             ).reshape(chi**3, chi**3)
  dtemp, vtemp = eigsh(0.5 * (ham_top + np.conj(ham_top.T)), k=1, which='SA')
  vtemp = vtemp / LA.norm(vtemp)
  v = vtemp.reshape(chi, chi, chi)

  # lower the density matrix
  rho[n_levels] = (vtemp @ np.conj(vtemp.T)).reshape([chi] * 6)
  for z in reversed(range(n_levels)):
    rho[z] = LowerDensity(u[z], w[z], rho[z + 1])

  # compute energy and magnetization
  energy_per_site = np.trace(rho[0].reshape(chi_b**3, chi_b**3) @ ham_init) / 2
  en_keep[p] = energy_per_site
  expectX = np.trace(rho[0].reshape(chi_b**3, chi_b**3) @
                     np.kron(np.eye(32), sX))
  energy_error = energy_per_site - energy_exact
  print('Iteration: %d of %d, Energy: %f, Energy Error: %e, XMag: %e\n'
        % (p, n_iterations, energy_per_site, energy_error, expectX))

  bias0 = tn.ncon([ham[0], rho[0]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
  ham[0] = ham[0] - (bias0 * np.eye(chi_b**3)).reshape([chi_b] * 6)


# plot results
plt.figure(1)
plt.yscale('log')
plt.xscale('log')
plt.plot(range(len(en_keep)), en_keep - energy_exact, 'b', label="MERA")
plt.plot(range(len(en_keep0)), en_keep0 - energy_exact, 'b', label="MERA")
plt.legend()
plt.title('MERA for XX model')
plt.xlabel('Update Step')
plt.ylabel('Ground Energy Error')
plt.show()

# np.save('XXData0.npy', (u, w, v, rho, ham, en_keep))
# u, w, v, rho, ham, en_keep = np.load('XXData0.npy', allow_pickle=True)

