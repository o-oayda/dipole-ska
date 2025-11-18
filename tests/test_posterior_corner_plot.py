import matplotlib

# Use a non-interactive backend for safety in tests.
matplotlib.use('Agg')

import numpy as np
import pytest

import dipoleska.utils.posterior as posterior_module
from dipoleska.utils.posterior import PosteriorMixin


class DummyPosterior(PosteriorMixin):
    def __init__(self, samples: np.ndarray, parameter_names: list[str]):
        self._samples = np.asarray(samples, dtype=np.float64)
        self._parameter_names = list(parameter_names)
        self._weighted_samples = None
        self._weights = None
        self._comparison_runs = []

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def parameter_names(self) -> list[str]:
        return self._parameter_names

    def model(self, Theta: np.ndarray):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def no_show(monkeypatch):
    monkeypatch.setattr(posterior_module.plt, 'show', lambda *_, **__: None)


def test_corner_plot_subsets_parameters(monkeypatch):
    calls = {}

    def fake_corner(samples, **kwargs):
        calls['samples'] = samples
        calls['labels'] = kwargs.get('labels')

    monkeypatch.setattr(posterior_module, 'corner', fake_corner)

    samples = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    params = ['phi', 'theta', 'Nbar']
    posterior = DummyPosterior(samples, params)

    posterior.corner_plot(
        backend='corner',
        parameters=['theta', 'Nbar']
    )

    assert calls['samples'].shape[1] == 2
    np.testing.assert_array_equal(
        calls['samples'],
        samples[:, [1, 2]]
    )
    assert calls['labels'] == ['$\\theta\\,(\\mathrm{rad})$', '$\\bar{N}_{\\mathrm{sources}}$']


def test_corner_plot_warns_on_incomplete_angles(monkeypatch):
    calls = {}

    def fake_corner(samples, **kwargs):
        calls['labels'] = kwargs.get('labels')

    monkeypatch.setattr(posterior_module, 'corner', fake_corner)

    samples = np.array([[0.1, 0.2, 0.3]])
    params = ['Nbar', 'D', 'phi']
    posterior = DummyPosterior(samples, params)

    with pytest.warns(RuntimeWarning, match='Skipping angle conversion'):
        posterior.corner_plot(
            backend='corner',
            coordinates=['equatorial'],
            parameters=params
        )

    assert calls['labels'] == ['$\\bar{N}_{\\mathrm{sources}}$', '$\\mathcal{D}_{\\mathrm{EB}}$', '$\\phi\\,(\\mathrm{rad})$']


def test_corner_plot_passes_selected_names_to_converter(monkeypatch):
    called = {}

    def fake_convert(
            self,
            samples,
            coordinates,
            parameter_names=None,
            return_conversion_flag=False
        ):
        called['parameter_names'] = parameter_names
        called['coordinates'] = coordinates
        result = np.full_like(samples, 42.0)
        return (result, True) if return_conversion_flag else result

    monkeypatch.setattr(
        DummyPosterior,
        '_convert_samples',
        fake_convert
    )
    monkeypatch.setattr(posterior_module, 'corner', lambda *_, **__: None)

    samples = np.array([[0.1, 0.2], [0.3, 0.4]])
    params = ['phi', 'theta']
    posterior = DummyPosterior(samples, params)

    posterior.corner_plot(
        backend='corner',
        coordinates=['equatorial'],
        parameters=['theta', 'phi']
    )

    assert called['parameter_names'] == ['theta', 'phi']
    assert called['coordinates'] == ['equatorial']


def test_corner_plot_allows_mismatched_comparison_with_subset(monkeypatch):
    def fake_corner(samples, **kwargs):
        return None

    monkeypatch.setattr(posterior_module, 'corner', fake_corner)

    base_samples = np.array([[0.1, 0.2, 0.3]])
    base_params = ['phi', 'theta', 'D']
    base = DummyPosterior(base_samples, base_params)

    comp_samples = np.array([[1.0, 2.0]])
    comp_params = ['theta', 'phi']
    comp = DummyPosterior(comp_samples, comp_params)
    base.add_comparison_run(comp, name='comp')

    # Should not raise; subset selects common angles
    base.corner_plot(
        backend='corner',
        parameters=['phi', 'theta']
    )


def test_corner_plot_mismatched_without_subset_errors(monkeypatch):
    monkeypatch.setattr(posterior_module, 'corner', lambda *_, **__: None)

    base = DummyPosterior(np.array([[0.1, 0.2]]), ['phi', 'theta'])
    comp = DummyPosterior(np.array([[0.3, 0.4]]), ['phi', 'theta', 'D'])
    base.add_comparison_run(comp, name='comp')

    with pytest.raises(ValueError, match='Provide a `parameters` list'):
        base.corner_plot(backend='corner')
