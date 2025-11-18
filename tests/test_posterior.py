import numpy as np
import pytest

from dipoleska.utils.posterior import PosteriorMixin
from dipoleska.utils.physics import change_source_coordinates


class DummyPosterior(PosteriorMixin):
    def __init__(self, samples: np.ndarray, parameter_names: list[str]):
        self._samples = np.asarray(samples, dtype=np.float64)
        self._parameter_names = parameter_names

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def parameter_names(self) -> list[str]:
        return self._parameter_names

    def model(self, Theta: np.ndarray):
        raise NotImplementedError


def _build_samples() -> tuple[DummyPosterior, np.ndarray]:
    parameter_names = [
        'A',
        'phi', 'theta',
        'phi_l2_0', 'theta_l2_0',
        'phi_l2_1', 'theta_l2_1'
    ]
    samples = np.array([
        [0.1, 0.0,   np.pi / 2, np.pi / 3, np.pi / 2, 3 * np.pi / 4, np.pi / 6],
        [0.2, np.pi, np.pi / 3, np.pi / 2, np.pi / 4, np.pi / 6,     np.pi / 2],
    ])
    posterior = DummyPosterior(samples, parameter_names)
    return posterior, samples


def test_convert_samples_to_degrees_without_rotation():
    posterior, base_samples = _build_samples()
    converted = posterior._convert_samples(base_samples, None)

    expected = base_samples.copy()
    for phi_idx, theta_idx in posterior._angle_parameter_pairs():
        phi = base_samples[:, phi_idx]
        theta = base_samples[:, theta_idx]
        expected[:, phi_idx] = np.degrees(phi)
        expected[:, theta_idx] = 90.0 - np.degrees(theta)

    np.testing.assert_allclose(converted, expected, atol=1e-9)


def test_convert_samples_with_rotation_to_galactic():
    posterior, base_samples = _build_samples()
    converted = posterior._convert_samples(
        base_samples,
        ['equatorial', 'galactic']
    )

    for phi_idx, theta_idx in posterior._angle_parameter_pairs():
        phi_deg = np.degrees(base_samples[:, phi_idx])
        lat_deg = 90.0 - np.degrees(base_samples[:, theta_idx])
        expected_lon, expected_lat = change_source_coordinates(
            phi_deg,
            lat_deg,
            native_coordinates='equatorial',
            target_coordinates='galactic'
        )
        np.testing.assert_allclose(
            converted[:, phi_idx],
            expected_lon,
            atol=1e-9
        )
        np.testing.assert_allclose(
            converted[:, theta_idx],
            expected_lat,
            atol=1e-9
        )


def test_convert_samples_rejects_invalid_coordinate_length():
    posterior, base_samples = _build_samples()
    with pytest.raises(ValueError):
        posterior._convert_samples(
            base_samples,
            ['equatorial', 'galactic', 'extra']
        )


def test_convert_samples_no_angle_parameters_returns_input():
    parameter_names = ['M0', 'M1']
    samples = np.array([[0.1, 0.2], [0.3, 0.4]])
    posterior = DummyPosterior(samples, parameter_names)
    converted = posterior._convert_samples(samples, None)
    np.testing.assert_allclose(converted, samples)


def test_add_comparison_run_stores_samples():
    posterior, base_samples = _build_samples()
    comparison_samples = base_samples * 0.5
    comparison = DummyPosterior(comparison_samples, posterior.parameter_names)
    comparison.name = 'CompRun'

    posterior.add_comparison_run(comparison)
    runs = posterior.comparison_runs

    assert len(runs) == 1
    assert runs[0].name == 'CompRun'
    np.testing.assert_allclose(runs[0].samples, comparison_samples)


def test_add_comparison_run_mismatched_names_warns():
    posterior, base_samples = _build_samples()
    other = DummyPosterior(base_samples, ['phi', 'theta'])

    with pytest.warns(RuntimeWarning):
        posterior.add_comparison_run(other)

    assert len(posterior.comparison_runs) == 1
