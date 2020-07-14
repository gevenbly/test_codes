
import numpy as np
from tensornetwork.block_sparse.blocksparsetensor import (BlockSparseTensor,
                                                          ChargeArray)
from tensornetwork.block_sparse.utils import (
    _find_transposed_diagonal_sparse_blocks, get_real_dtype)
from typing import (Tuple, Optional, Text)


def eigh(matrix: BlockSparseTensor,
         link_charges: Optional[ChargeArray] = None,
         which: Optional[Text] = 'LM',
         full_sort: Optional[bool] = True,
         threshold: Optional[float] = 0,
         UPLO: Optional[Text] = 'L'
         ) -> Tuple[ChargeArray, BlockSparseTensor]:
  """
  Compute eigenvectors and eigenvalues of a hermitian matrix (or array that can
  be reshaped into a hermitian matrix). Produces only a restricted set of
  eigenvalues with charges in ChargeArray `link_charges` if provided.
  Args:
    matrix: a `BlockSparseTensor` hermitian matrix, or an array that can
      become a hermitian matrix when reshaped between first N/2 and remaining
      N/2 indices.
    link_charges: an optional `ChargeArray` describing the charges of the
      eigenvalues to compute.
    which: optional str [‘LM’ | ‘SM’ | ‘LA’ | ‘SA’] describing how eigenvalues
      are be ordered.
    full_sort: if True then eigenvalues from are sorted together (regardless of
      their charge), else eigenvalues are only sorted within their charge
      blocks.
    threshold: if provided then all eigenvalues smaller than the threshold are
      automatically truncated.

  Returns:
    (ChargeArray,BlockSparseTensor): The eigenvalues and eigenvectors
  """

  if link_charges is None:
    return _eigh_free(matrix,
                      which=which,
                      full_sort=full_sort,
                      threshold=threshold,
                      UPLO=UPLO)
  else:
    return _eigh_fixed(matrix,
                       link_charges=link_charges,
                       which=which,
                       UPLO=UPLO)


def _eigh_free(matrix: BlockSparseTensor,
               which: Optional[Text] = 'LM',
               full_sort: Optional[bool] = True,
               threshold: Optional[float] = 0,
               UPLO: Optional[Text] = 'L'
               ) -> Tuple[ChargeArray, BlockSparseTensor]:
  """
  Compute eigenvectors and eigenvalues of a hermitian matrix where the output
  charge order is free.
  """

  # reshape into matrix if needed
  pivot = matrix.ndim // 2
  m_shape = matrix.shape
  matrix = matrix.reshape([np.prod(m_shape[:pivot]), np.prod(m_shape[pivot:])])

  # compute info about each block
  flat_charges = matrix._charges
  flat_flows = matrix._flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  # diagonalize each block
  eigvals = []
  v_blocks = []
  for n, block in enumerate(blocks):
    e, v = np.linalg.eigh(np.reshape(matrix.data[block], shapes[:, n]), UPLO)

    # discard e-values and vectors that fall below threshold
    if threshold > 0:
      locs = abs(e) >= threshold
      e = e[locs]
      v = v[:, locs]

    if full_sort:
      eigvals.append(e)
      v_blocks.append(v.T)
    else:
      # sort within each block
      if which == 'SA':
        blk_sort = np.argsort(e)
      elif which == 'LA':
        blk_sort = np.flip(np.argsort(e))
      elif which == 'SM':
        blk_sort = np.argsort(np.abs(e))
      elif which == 'LM':
        blk_sort = np.flip(np.argsort(np.abs(e)))

      eigvals.append(e[blk_sort])
      v_blocks.append(v[:, blk_sort].T)

  # combine e-values from all blocks
  if len(eigvals) > 0:
    all_eigvals = np.concatenate(eigvals)
  else:
    all_eigvals = np.empty(0, dtype=get_real_dtype(matrix.dtype))

  if full_sort:
    # do global sort of all e-values
    if which == 'SA':
      glb_sort = np.argsort(all_eigvals)
    elif which == 'LA':
      glb_sort = np.flip(np.argsort(all_eigvals))
    elif which == 'SM':
      glb_sort = np.argsort(np.abs(all_eigvals))
    elif which == 'LM':
      glb_sort = np.flip(np.argsort(np.abs(all_eigvals)))
  else:
    glb_sort = np.arange(len(all_eigvals), dtype=np.int64)

  # make charge labels and block positions of each e-value
  tmp_labels = [np.full(len(eigvals[n]), fill_value=n, dtype=np.int16)
                for n in range(len(eigvals))]
  tmp_degens = [np.arange(len(eigvals[n]), dtype=np.int16)
                for n in range(len(eigvals))]
  if len(tmp_labels) > 0:
    e_labels = np.concatenate(tmp_labels)[glb_sort]
    e_degens = np.concatenate(tmp_degens)[glb_sort]
  else:
    e_labels = np.empty(0, dtype=np.int16)
    e_degens = np.empty(0, dtype=np.int16)

  e_charge = charges[e_labels]
  E = ChargeArray(all_eigvals[glb_sort], [e_charge], [False])

  charges_v = [e_charge] + [matrix._charges[o] for o in matrix._order[0]]
  order_v = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
  flows_v = [True] + [matrix._flows[o] for o in matrix._order[0]]

  if len(v_blocks) > 0:
    all_v_blocks = np.concatenate([v_blocks[e_labels[n]][e_degens[n], :]
                                   for n in range(len(all_eigvals))])
  else:
    all_v_blocks = np.empty(0, dtype=matrix.dtype)

  fin_shape = [*m_shape[:pivot], len(all_eigvals)]
  V = BlockSparseTensor(
      all_v_blocks,
      charges=charges_v,
      flows=flows_v,
      order=order_v,
      check_consistency=False).transpose().reshape(fin_shape)

  return E, V


def _eigh_fixed(matrix: BlockSparseTensor,
                link_charges: ChargeArray,
                which: Optional[Text] = 'LM',
                UPLO: Optional[Text] = 'L'
                ) -> Tuple[ChargeArray, BlockSparseTensor]:
  """
  Compute eigenvectors and eigenvalues of a hermitian matrix where the output
  charge order is fixed to match that of `link_charges`.
  """

  # reshape into matrix if needed
  pivot = matrix.ndim // 2
  m_shape = matrix.shape
  matrix = matrix.reshape([np.prod(m_shape[:pivot]), np.prod(m_shape[pivot:])])

  # compute info about each block
  flat_charges = matrix._charges
  flat_flows = matrix._flows
  flat_order = matrix.flat_order
  tr_partition = len(matrix._order[0])
  blocks, charges, shapes = _find_transposed_diagonal_sparse_blocks(
      flat_charges, flat_flows, tr_partition, flat_order)

  # intersect between link charges and block charges
  link_uni, link_pos, link_counts = link_charges.unique(return_inverse=True,
                                                        return_counts=True)
  _, blk_common, link_common = charges.intersect(link_uni,
                                                 return_indices=True)

  # diagonalize each block
  eigvals = []
  v_blocks = []
  for n, m in enumerate(blk_common):
    e, v = np.linalg.eigh(np.reshape(matrix.data[blocks[m]],
                                     shapes[:, m]), UPLO)

    # sort within each block
    if which == 'SA':
      blk_sort = np.argsort(e)
    elif which == 'LA':
      blk_sort = np.flip(np.argsort(e))
    elif which == 'SM':
      blk_sort = np.argsort(np.abs(e))
    elif which == 'LM':
      blk_sort = np.flip(np.argsort(np.abs(e)))

    eigvals.append(e[blk_sort])
    v_blocks.append(v[:, blk_sort].T)

  link_degens = np.zeros(np.size(link_pos), dtype=np.int64)
  for n, count in enumerate(link_counts):
    link_degens[link_pos == n] = np.arange(count, dtype=np.int64)

  all_eigvals = np.zeros(link_pos.size, dtype=get_real_dtype(matrix.dtype))
  for n in range(len(link_pos)):
    all_eigvals[n] = eigvals[link_pos[n]][link_degens[n]]

  e_charge = charges[blk_common[link_pos]]
  E = ChargeArray(all_eigvals, [e_charge], [False])

  charges_v = [e_charge] + [matrix._charges[o] for o in matrix._order[0]]
  order_v = [[0]] + [list(np.arange(1, len(matrix._order[0]) + 1))]
  flows_v = [True] + [matrix._flows[o] for o in matrix._order[0]]

  if len(v_blocks) > 0:
    all_v_blocks = np.concatenate([v_blocks[link_pos[n]][link_degens[n], :]
                                   for n in range(len(all_eigvals))])
  else:
    all_v_blocks = np.empty(0, dtype=matrix.dtype)

  fin_shape = [*m_shape[:pivot], len(all_eigvals)]
  V = BlockSparseTensor(
      all_v_blocks,
      charges=charges_v,
      flows=flows_v,
      order=order_v,
      check_consistency=False).transpose().reshape(fin_shape)

  return E, V
