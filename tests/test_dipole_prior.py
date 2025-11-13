import healpy as hp
import numpy as np
import pytest

from dipoleska.models.dipole import Dipole
from dipoleska.models.priors import Prior


def _simple_density_map(nside: int = 1) -> np.ndarray:
    return np.full(hp.nside2npix(nside), 10.0, dtype=float)


def test_default_point_prior_contains_only_dipole_angles() -> None:
    density_map = _simple_density_map()

    model = Dipole(density_map=density_map, prior=None, likelihood='point')

    assert model.prior.parameter_names == ['D', 'phi', 'theta']


def test_custom_prior_overrides_default_bounds() -> None:
    density_map = _simple_density_map()
    custom = Prior(choose_prior={'phi': ['Uniform', 0.5, 0.75]})

    model = Dipole(density_map=density_map, prior=custom, likelihood='poisson')

    prior_dict = model.prior.prior_dict
    assert prior_dict['phi'][1:] == [0.5, 0.75]
    assert prior_dict['Nbar'][1:] == pytest.approx([0.75 * 10.0, 1.25 * 10.0])


def test_unknown_prior_name_raises() -> None:
    density_map = _simple_density_map()
    bad_prior = Prior(choose_prior={'unknown': ['Uniform', 0.0, 1.0]})

    with pytest.raises(ValueError, match='Unrecognised prior parameter'):
        Dipole(density_map=density_map, prior=bad_prior, likelihood='poisson')


def test_default_parameters_are_announced(capsys: pytest.CaptureFixture[str]) -> None:
    density_map = _simple_density_map()
    custom = Prior(choose_prior={'D': ['Uniform', 0.0, 0.2]})

    Dipole(density_map=density_map, prior=custom, likelihood='poisson')

    message = capsys.readouterr().out
    assert 'Nbar' in message
