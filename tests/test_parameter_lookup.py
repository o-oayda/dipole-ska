import healpy as hp
import numpy as np
import pytest

from dipoleska.models.dipole import Dipole
from dipoleska.models.multipole import Multipole
from dipoleska.models.priors import Prior


def _constant_map(nside: int = 1, value: float = 10.0) -> np.ndarray:
    return np.full(hp.nside2npix(nside), value, dtype=float)


def test_prior_index_lookup_updates_after_add_and_remove() -> None:
    prior = Prior(choose_prior={'A': ['Uniform', 0.0, 1.0], 'B': ['Uniform', 0.0, 2.0]})
    assert prior.index_for('A') == 0
    assert prior.index_for('B') == 1

    prior.add_prior(prior_index=1, prior_name='C', prior_alias=['Uniform', 0.0, 3.0])
    assert prior.index_for('C') == 1
    assert prior.index_for('B') == 2

    prior.remove_prior([1])
    assert prior.index_for('B') == 1
    with pytest.raises(KeyError):
        prior.index_for('C')


def test_dipole_model_uses_named_parameter_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    density_map = _constant_map()
    model = Dipole(density_map=density_map, likelihood='point')

    recorded: dict[str, np.ndarray] = {}

    def fake_signal(*, dipole_amplitude, dipole_longitude, dipole_colatitude, pixel_vectors):
        recorded['amplitude'] = dipole_amplitude.copy()
        recorded['phi'] = dipole_longitude.copy()
        recorded['theta'] = dipole_colatitude.copy()
        return np.zeros((pixel_vectors.shape[0], dipole_amplitude.shape[0]))

    monkeypatch.setattr('dipoleska.models.dipole.compute_dipole_signal', fake_signal)

    # Simulate alternative column ordering.
    model._parameter_indices = {'phi': 0, 'D': 1, 'theta': 2}
    Theta = np.array([[1.2, 0.3, 0.7]])

    output = model.model(Theta)

    np.testing.assert_allclose(recorded['amplitude'], [0.3])
    np.testing.assert_allclose(recorded['phi'], [1.2])
    np.testing.assert_allclose(recorded['theta'], [0.7])
    np.testing.assert_allclose(output, 1.0)


def test_multipole_model_uses_named_parameter_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    density_map = _constant_map()
    model = Multipole(density_map=density_map, ells=[1])

    recorded: dict[str, np.ndarray] = {}

    def fake_signal(*, dipole_amplitude, dipole_longitude, dipole_colatitude, pixel_vectors):
        recorded['amplitude'] = dipole_amplitude.copy()
        recorded['phi'] = dipole_longitude.copy()
        recorded['theta'] = dipole_colatitude.copy()
        return np.zeros((pixel_vectors.shape[0], dipole_amplitude.shape[0]))

    monkeypatch.setattr('dipoleska.models.multipole.compute_dipole_signal', fake_signal)

    # Re-map indices so phi/theta/amplitude live at custom locations.
    model._parameter_indices = {
        'phi_l1_0': 0,
        'theta_l1_0': 1,
        'M1': 2,
    }
    model.phi_indices = {'1': [0]}
    model.theta_indices = {'1': [1]}

    Theta = np.array([[0.5, 0.9, 0.2]])
    output = model.model(Theta)

    np.testing.assert_allclose(recorded['amplitude'], [0.2])
    np.testing.assert_allclose(recorded['phi'], [0.5])
    np.testing.assert_allclose(recorded['theta'], [0.9])
    np.testing.assert_allclose(output, 1.0)
