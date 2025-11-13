import matplotlib.pyplot as plt
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

plt.scatter(masked_rmsmap, masked_dmap, s=1)
plt.show()

model = Dipole(masked_dmap, likelihood='general_poisson_rms', rms_map=masked_rmsmap)
model.prior.plot_priors()

model.run_nested_sampling()
model.corner_plot(coordinates=['equatorial', 'galactic'])
model.sky_direction_posterior(coordinates=['equatorial', 'galactic'])
plt.show()
