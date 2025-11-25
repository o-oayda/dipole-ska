#%%
from dipoleska.utils.modulated_map_generator import ModulatedMapGenerator
from dipoleska.powerspectrum.power_spectrum_plotter import LegacyPowerSpectrumPlotter
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapLoader
from dipoleska.models.multipole import Multipole
from dipoleska.powerspectrum.posterior_power_spectrum import PosteriorPowerSpectrum
import matplotlib.pyplot as plt
#%%

LIKELIHOOD = 'general_poisson'

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
processor.mask(
    classification=['north_equatorial', 'south_equatorial', 'galactic_plane'],
    radius=[10,85,10],
    output_frame='G'
)
density_map_dscaled = processor.density_map
model = Multipole(density_map_dscaled, likelihood=LIKELIHOOD, ells=[0, 1])
model.run_nested_sampling(step=True)
#%%

## create a new model without the mask. We need to calculate the power spectrum
## for the full sky reconstruction to see what the NS run is anticipating.
## this is done only to remove info about the mask from the model.
loader2 = MapLoader(-1, 'AA')
density_map2 = loader2.load(101)
modulator2 = ModulatedMapGenerator(
        density_map2,
        dipole_amplitude=0.05,
        dipole_longitude=264,
        dipole_latitude=48
    )
dipole_map2 = modulator2.modulated_map(scaling_factor=100)
processor2 = MapProcessor(dipole_map2)
processor2.change_map_resolution(nside_out=64)
density_map_dscaled2 = processor2.density_map
model2 = Multipole(density_map_dscaled2, likelihood=LIKELIHOOD, ells=[0, 1])

# Calculate the posterior power spectrum from the model samples.
# Note that we need power spectrum for same nside as the SKA maps.
posteriorps = PosteriorPowerSpectrum(
    sample_chains=model.samples,
    model=model2.model,
    likelihood=LIKELIHOOD,
    sample_count=500
)
cl_mean, cl_std = posteriorps.power_spectrum_calculator()
#%%
# Plot the power spectra including the posterior power spectrum.
ca, cf = LegacyPowerSpectrumPlotter(101, 150, -1, 'AA').plot_power_spectra()
ca.errorbar(1, cl_mean[1], yerr=cl_std[1], fmt='o', color='red', label='Sampled $C_l$ at $l=1$',elinewidth=2)
ca.errorbar(2, cl_mean[2], yerr=cl_std[2], fmt='o', color='blue', label='Sampled $C_l$ at $l=2$',elinewidth=2)
ca.set_ylim(1e-8, 1e-2)
ca.set_xlim(0.9,len(cl_mean))
ca.legend()

plt.show()
