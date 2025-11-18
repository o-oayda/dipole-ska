import matplotlib

# Use a non-interactive backend to avoid GUI requirements during tests.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dipoleska.utils.plotting import MapPlotter
import healpy as hp


def _stub_projview(monkeypatch):
    """
    Replace healpy.projview with a stub that respects the 'sub' kwarg to create
    distinct matplotlib axes in line with the requested subplot index.
    """
    import dipoleska.utils.plotting as plotting

    calls = []

    def _projview_stub(map_data, cmap=None, **kwargs):
        fig_arg = kwargs.get('fig')
        if fig_arg is not None:
            plt.figure(fig_arg)
        if 'sub' in kwargs:
            plt.subplot(*kwargs['sub'])
        else:
            plt.gca()
        calls.append({'map': map_data, 'kwargs': kwargs})
        return plt.gca()

    monkeypatch.setattr(plotting.hp, 'projview', _projview_stub)
    return calls


def test_accepts_multiple_maps_with_different_sizes():
    # Two valid HEALPix-sized arrays with different sizes should be accepted.
    map_small = np.zeros(hp.nside2npix(1))
    map_large = np.ones(hp.nside2npix(2))

    plotter = MapPlotter([map_small, map_large])

    assert len(plotter.density_maps) == 2
    assert plotter.density_map is map_small


def test_rejects_non_array_entries():
    with pytest.raises(TypeError):
        MapPlotter([np.zeros(hp.nside2npix(1)), "not-an-array"])


def test_plot_density_map_with_subplots(monkeypatch):
    calls = _stub_projview(monkeypatch)
    map_one = np.arange(hp.nside2npix(1))
    map_two = np.arange(hp.nside2npix(2))

    plotter = MapPlotter([map_one, map_two])
    axes, _ = plotter.plot_density_map()

    assert len(calls) == 2
    assert calls[0]['kwargs'].get('sub') == (1, 2, 1)
    assert calls[1]['kwargs'].get('sub') == (1, 2, 2)
    assert calls[0]['kwargs'].get('fig') == calls[1]['kwargs'].get('fig')
    assert isinstance(axes, list)
    assert len(axes) == 2


def test_plot_smooth_map_uses_smoothed_maps(monkeypatch):
    calls = _stub_projview(monkeypatch)
    map_one = np.zeros(hp.nside2npix(1))
    map_two = np.ones(hp.nside2npix(1))

    # Avoid heavy healpy operations in smoothing by stubbing.
    def _fake_smooth(map_data, **_kwargs):
        return np.full_like(map_data, 5.0, dtype=float)

    monkeypatch.setattr(MapPlotter, 'moving_average_smooth', staticmethod(_fake_smooth))

    plotter = MapPlotter([map_one, map_two])
    axes, _ = plotter.plot_smooth_map()

    assert len(calls) == 2
    np.testing.assert_allclose(plotter.smoothed_maps[0], 5.0)
    np.testing.assert_allclose(plotter.smoothed_maps[1], 5.0)
    assert plotter.smoothed_map is plotter.smoothed_maps[0]
    assert isinstance(axes, list)
    assert len(axes) == 2
    assert calls[0]['kwargs'].get('fig') == calls[1]['kwargs'].get('fig')
