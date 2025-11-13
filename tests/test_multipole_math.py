import itertools

import numpy as np

from dipoleska.utils.math import make_rankn_traceless, symmetrise_tensor


def test_symmetrise_tensor_matches_manual_average() -> None:
    tensor = np.arange(27, dtype=float).reshape(3, 3, 3)
    perms = list(itertools.permutations(range(tensor.ndim)))
    manual_average = sum(np.transpose(tensor, perm) for perm in perms) / len(perms)

    symmetrised = symmetrise_tensor(tensor)

    np.testing.assert_allclose(symmetrised, manual_average)
    for perm in perms:
        np.testing.assert_allclose(symmetrised, np.transpose(symmetrised, perm))


def test_make_rankn_traceless_is_symmetric_and_traceless() -> None:
    rng = np.random.default_rng(1234)
    tensor = rng.normal(size=(3, 3, 3, 3))
    sym_tensor = symmetrise_tensor(tensor)

    traceless_tensor = make_rankn_traceless(sym_tensor)

    # Symmetry: invariant under any permutation of indices.
    perms = list(itertools.permutations(range(traceless_tensor.ndim)))
    for perm in perms:
        np.testing.assert_allclose(
            traceless_tensor,
            np.transpose(traceless_tensor, perm),
            atol=1e-12,
        )

    # Traceless: contraction over any pair of indices vanishes.
    rank = traceless_tensor.ndim
    for axis1 in range(rank):
        for axis2 in range(axis1 + 1, rank):
            contracted = np.trace(traceless_tensor, axis1=axis1, axis2=axis2)
            assert np.allclose(contracted, 0.0, atol=1e-10)
