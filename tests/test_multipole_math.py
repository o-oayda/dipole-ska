import functools
import itertools

import numpy as np

from dipoleska.utils.math import (
    make_rankn_traceless,
    make_vectorised_outer_einstring,
    make_vectorised_signal_einstring,
    multipole_pixel_product_vectorised,
    multipole_tensor_vectorised,
    symmetrise_tensor,
    vectorised_outer_product,
    vectorised_quadrupole_tensor,
)


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


def test_make_vectorised_outer_einstring_increments_letters() -> None:
    assert make_vectorised_outer_einstring(1) == 'a...'
    assert make_vectorised_outer_einstring(4) == 'a...,b...,c...,d...'


def test_make_vectorised_signal_einstring_layout() -> None:
    assert make_vectorised_signal_einstring(1) == 'ab,b...'
    assert make_vectorised_signal_einstring(3) == 'abcd,b...,c...,d...'


def test_vectorised_outer_product_matches_manual_outer() -> None:
    n_live = 2
    ell = 3
    cartesian_vectors = np.arange(3 * n_live * ell, dtype=float).reshape(3, n_live, ell) / 5

    tensor = vectorised_outer_product(cartesian_vectors)

    expected = np.empty_like(tensor)
    for sample in range(n_live):
        vectors = [cartesian_vectors[:, sample, idx] for idx in range(ell)]
        manual = functools.reduce(np.multiply.outer, vectors)
        expected[sample] = manual

    np.testing.assert_allclose(tensor, expected)


def test_multipole_tensor_vectorised_matches_manual_pipeline() -> None:
    ell = 2
    n_live = 2
    amplitudes = np.array([0.5, 2.0])
    cart_vectors = np.zeros((3, n_live, ell))
    cart_vectors[:, 0, 0] = [1.0, 0.0, 0.0]
    cart_vectors[:, 0, 1] = [0.0, 1.0, 0.0]
    cart_vectors[:, 1, 0] = [0.0, 0.0, 1.0]
    cart_vectors[:, 1, 1] = [1.0, 1.0, 1.0]

    tensor = multipole_tensor_vectorised(amplitudes, cart_vectors)

    expected = []
    for idx in range(n_live):
        vectors = [cart_vectors[:, idx, j] for j in range(ell)]
        raw = functools.reduce(np.multiply.outer, vectors)
        sym = symmetrise_tensor(raw)
        stf = make_rankn_traceless(sym)
        expected.append(amplitudes[idx] * stf)
    expected = np.asarray(expected)

    np.testing.assert_allclose(tensor, expected)


def test_multipole_pixel_product_vectorised_matches_manual() -> None:
    ell = 2
    multipole_tensors = np.arange(18, dtype=float).reshape(2, 3, 3) / 7
    pixel_vectors = np.arange(12, dtype=float).reshape(3, 4) / 3

    signal = multipole_pixel_product_vectorised(multipole_tensors, pixel_vectors, ell=ell)

    n_pix = pixel_vectors.shape[1]
    n_live = multipole_tensors.shape[0]
    expected = np.zeros((n_pix, n_live))
    for pix in range(n_pix):
        vec = pixel_vectors[:, pix]
        for sample in range(n_live):
            expected[pix, sample] = vec @ multipole_tensors[sample] @ vec

    np.testing.assert_allclose(signal, expected)


def test_vectorised_quadrupole_tensor_symmetry_and_tracelessness() -> None:
    amplitudes = np.array([1.5, 0.5])
    cart_vectors = np.zeros((3, 2, 2))
    cart_vectors[:, 0, 0] = [1.0, 0.0, 0.0]
    cart_vectors[:, 0, 1] = [0.0, 1.0, 0.0]
    cart_vectors[:, 1, 0] = [0.0, 0.0, 1.0]
    cart_vectors[:, 1, 1] = [1.0, 1.0, 1.0] / np.sqrt(3)

    tensors = vectorised_quadrupole_tensor(
        amplitude_like=amplitudes,
        cartesian_quadrupole_vectors=cart_vectors
    )

    assert tensors.shape == (2, 3, 3)

    for tensor in tensors:
        np.testing.assert_allclose(tensor, tensor.T, atol=1e-12)
        np.testing.assert_allclose(np.trace(tensor), 0.0, atol=1e-12)

    # Validate against direct symmetrisation + traceless projection.
    expected = []
    for sample in range(amplitudes.shape[0]):
        v1 = cart_vectors[:, sample, 0]
        v2 = cart_vectors[:, sample, 1]
        raw = np.outer(v1, v2)
        sym = (raw + raw.T) / 2
        stf = sym - (np.trace(sym) / 3) * np.eye(3)
        expected.append(amplitudes[sample] * stf)
    expected = np.asarray(expected)

    np.testing.assert_allclose(tensors, expected)


def test_multipole_tensor_vectorised_rank3_matches_manual_pipeline() -> None:
    ell = 3
    n_live = 1
    amplitudes = np.array([1.23])

    cart_vectors = np.zeros((3, n_live, ell))
    cart_vectors[:, 0, 0] = [1.0, 0.0, 0.0]
    cart_vectors[:, 0, 1] = [0.0, 1.0, 0.0]
    cart_vectors[:, 0, 2] = [0.0, 0.0, 1.0]

    tensor = multipole_tensor_vectorised(amplitudes, cart_vectors)[0]

    vectors = [cart_vectors[:, 0, idx] for idx in range(ell)]
    raw = functools.reduce(np.multiply.outer, vectors)
    sym = symmetrise_tensor(raw)
    expected = amplitudes[0] * make_rankn_traceless(sym)

    np.testing.assert_allclose(tensor, expected)
