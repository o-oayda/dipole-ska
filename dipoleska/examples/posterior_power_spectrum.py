#%%
from dipoleska.utils.modulated_map_generator import ModulatedMapGenerator
from dipoleska.powerspectrum.power_spectrum_plotter import PowerSpectrumPlotter
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapLoader
from dipoleska.models.multipole import Multipole
from dipoleska.powerspectrum.posterior_power_spectrum import PosteriorPowerSpectrum
import healpy as hp
import numpy as np
#%%
# Load a SKA map, modulate it with a dipole, process it, and run multipole model.
loader = MapLoader(-1, 'AA')
density_map = loader.load(101)
modulator = ModulatedMapGenerator(
        density_map,
        dipole_amplitude=0.05,
        dipole_longitude=264,
        dipole_latitude=48
    )
dipole_map = modulator.modulated_map(scaling_factor=100)
processor = MapProcessor(dipole_map)
processor.change_map_resolution(nside_out=64)
density_map_dscaled = processor.density_map
model = Multipole(density_map_dscaled, ells=[1, 2])
model.run_nested_sampling(step=True)
#%%
# Calculate the posterior power spectrum from the model samples.
# Note that we need power spectrum for same nside as the SKA maps.
model_for_power_spectrum = Multipole(density_map, ells=[1, 2])
posteriorps = PosteriorPowerSpectrum(model._samples,
                                     model_for_power_spectrum.model,
                                     sample_count=500)
cl_mean, cl_std = posteriorps.power_spectrum_calculator()
#%%
# Plot the power spectra including the posterior power spectrum.
ca, cf = PowerSpectrumPlotter(101, 150, -1, 'AA').plot_power_spectra()
ca.errorbar(1, cl_mean[1], yerr=cl_std[1], fmt='o', color='red', label='Sampled $C_l$ at $l=1$',elinewidth=2)
ca.errorbar(2, cl_mean[2], yerr=cl_std[2], fmt='o', color='blue', label='Sampled $C_l$ at $l=2$',elinewidth=2)
ca.set_ylim(1e-8, 1e-2)
ca.set_xlim(0.9,len(cl_mean))
ca.legend()