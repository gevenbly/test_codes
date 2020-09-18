

# general python modules
import numpy as np

# tensornetwork and block_sparse modules
from tensornetwork.block_sparse.charge import Z2Charge
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse import BlockSparseTensor as BT
from tensornetwork.block_sparse import linalg as BLA
import tensornetwork as tn

# tensornetwork and block_sparse modules
from eigh import eigh

tn.set_default_backend('symmetric')

"""
Example 1: diagonalize Ising model for the ground state within each symmetry
sector using block_sparse with Z2 symmetry.
"""
# model parameters
n_sites = 10
mag = 1.1

# solve analytic ground energy
m1 = np.arange((1 - n_sites) / 2, (n_sites + 1) / 2, 1)
q1 = 2 * np.pi * m1 / n_sites
wq1 = np.sqrt(1 + 2 * mag * np.cos(q1) + mag**2)

m2 = np.arange(-n_sites / 2, n_sites / 2, 1)
q2 = 2 * np.pi * m2 / n_sites
wq2 = np.sqrt(1 + 2 * mag * np.cos(q2) + mag**2)

en_even0 = -sum(wq1)
if mag > 1:
  en_odd0 = -sum(wq2) + 2 * wq2[0]
else:
  en_odd0 = -sum(wq2)

# define Ising hamiltonian
q_chib = Z2Charge([0, 1])
ind_chib0 = Index(charges=q_chib, flow=True)
ind_chib1 = Index(charges=q_chib, flow=False)

sX = np.array([[0, 1], [1, 0]])
sZ = np.array([[1, 0], [0, -1]])
ham_s = (-np.kron(sX, sX) + 0.5 * mag * np.kron(np.eye(2), sZ) +
         0.5 * mag * np.kron(np.eye(2), sZ))
ham_init = (np.kron(ham_s, np.eye(2**(n_sites - 2)))
            ).reshape(2 * np.ones(2 * n_sites, dtype=int))

ham_Z0 = BT.fromdense([ind_chib1] * n_sites + [ind_chib0] * n_sites, ham_init)

cyc_perm = np.array([*list(range(1, n_sites)), 0])
ham_temp = ham_Z0
ham_final = ham_Z0
for n in range(n_sites - 1):
  ham_temp = ham_temp.transpose([*cyc_perm, *(cyc_perm + n_sites)])
  ham_final = ham_final + ham_temp

# diagonalize Ising hamiltonian for GS in even and odd symmetry sector
E, V = eigh(ham_final, link_charges=Z2Charge([0, 1]), which='SA')
en_even1 = E.todense()[0].item()
en_odd1 = E.todense()[1].item()

# compare with analytic energies
assert np.allclose(en_even0, en_even1)
assert np.allclose(en_odd0, en_odd1)


"""
Example 2: compute truncated eigendecomposition of a reduced density matrix,
keeping only the eigenvalues above some cut-off threshold
"""

rho_temp = BT.fromdense([ind_chib1] + [ind_chib0],
                        np.array([[1, 0], [0, 0]], dtype=float))
V = V.reshape([2**(n_sites // 2), 2**(n_sites // 2), 2])
rho_half = tn.ncon([V, rho_temp, V.conj()], [[-1, 1, 2], [2, 3], [-2, 1, 3]])

# decomp with evalues sorted by magnitude
E2, V2 = eigh(rho_half, which='LM', full_sort=False, threshold=1e-10, max_kept=15)
rho_recover = V2 @ BLA.diag(E2) @ V2.T.conj()
assert np.allclose(rho_half.todense(), rho_recover.todense())

# decomp with evalues sorted by magnitude within each charge block
E2, V2 = eigh(rho_half, which='LM', threshold=1e-10, full_sort=False)
rho_recover = V2 @ BLA.diag(E2) @ V2.T.conj()
assert np.allclose(rho_half.todense(), rho_recover.todense())

# decomp with no truncation
E2, V2 = eigh(rho_half, which='LM')
rho_recover = V2 @ BLA.diag(E2) @ V2.T.conj()
assert np.allclose(rho_half.todense(), rho_recover.todense())
