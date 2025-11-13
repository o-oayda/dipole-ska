import healpy as hp
import numpy as np
import pytest

from dipoleska.models.multipole import Multipole
from dipoleska.models.priors import Prior


def _simple_density_map(nside: int = 1) -> np.ndarray:
    return np.full(hp.nside2npix(nside), 12.0, dtype=float)


def test_default_prior_for_ells_includes_expected_parameters() -> None:
    density_map = _simple_density_map()

    model = Multipole(density_map=density_map, ells=[0, 1])

    expected = {'M0', 'M1', 'phi_l1_0', 'theta_l1_0'}
    assert set(model.prior.parameter_names) == expected


def test_multipole_prior_orders_amplitudes_before_angles() -> None:
    density_map = _simple_density_map()

    model = Multipole(density_map=density_map, ells=[0, 1, 2])

    assert model.prior.parameter_names == [
        'M0',
        'M1',
        'M2',
        'phi_l1_0',
        'theta_l1_0',
        'phi_l2_0',
        'theta_l2_0',
        'phi_l2_1',
        'theta_l2_1',
    ]


def test_custom_prior_overrides_default_and_keeps_others() -> None:
    density_map = _simple_density_map()
    custom = Prior(choose_prior={'M1': ['Uniform', 0.0, 0.05]})

    model = Multipole(density_map=density_map, ells=[1], prior=custom)

    prior_dict = model.prior.prior_dict
    assert prior_dict['M1'][2] == 0.05
    assert prior_dict['phi_l1_0'] == ['Uniform', 0.0, 2 * np.pi]


def test_unknown_prior_name_raises_for_multipole() -> None:
    density_map = _simple_density_map()
    bad_prior = Prior(choose_prior={'bogus': ['Uniform', 0.0, 1.0]})

    with pytest.raises(ValueError, match='Unrecognised prior parameter'):
        Multipole(density_map=density_map, ells=[1], prior=bad_prior)


def test_default_notice_printed_when_needed(capsys: pytest.CaptureFixture[str]) -> None:
    density_map = _simple_density_map()
    custom = Prior(choose_prior={'phi_l1_0': ['Uniform', 0.0, 1.0]})

    Multipole(density_map=density_map, ells=[1], prior=custom)

    message = capsys.readouterr().out
    assert 'M1' in message
