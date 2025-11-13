"""
Custom prior example.

This script demonstrates how to override only the parameters you care about
while letting the Dipole model supply defaults for the rest. The model builds
its own priors when ``prior=None``; supplying a ``Prior`` instance simply
replaces any matching parameter names. In this example we tighten the dipole
amplitude (`D`) while leaving `Nbar`, `rms_slope`, `phi`, and `theta` at their
defaults.

To discover the recognised parameter names (e.g. `D`, `phi`, `theta`,
`Nbar`, `rms_slope`, `gp_dispersion`), check the Dipole docstring in
``dipoleska/models/dipole.py`` or inspect ``model.prior.parameter_names`` after
initialisation. The same pattern holds for the Multipole model, whose names are
listed in ``dipoleska/models/multipole.py``.
"""

from dipoleska.models.priors import Prior
from dipoleska.models.dipole import Dipole
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapCollectionLoader


loader = MapCollectionLoader(
    snr_cut=10, 
    lower_flux_limit='5e-4',
    lower_z_limit='0.5',
    gal_cut=10,
    map_types=['all']
)
data = loader.map_collections

dmap = data['counts']
processor = MapProcessor(dmap)
processor.mask(output_frame='C', load_from_file='gal10_ps')
masked_dmap = processor.density_map

rmsmap = data['rms']
processor = MapProcessor(rmsmap)
processor.mask(output_frame='C', load_from_file='gal10_ps')
masked_rmsmap = processor.density_map

prior = Prior(
    choose_prior={
        'D': ['Uniform', 0., 0.2],
    }
)

model = Dipole(masked_dmap, prior=prior, likelihood='poisson_rms', rms_map=masked_rmsmap)
model.prior.plot_priors()
model.run_nested_sampling()
model.corner_plot(coordinates=['equatorial', 'galactic'])
