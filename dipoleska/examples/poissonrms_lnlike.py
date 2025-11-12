import matplotlib.pyplot as plt
from dipoleska.models.dipole import Dipole
from dipoleska.models.multipole import Multipole
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapCollectionLoader
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    choices=['dipole', 'multipole'],
    default='dipole',
    help='Choose which model to fit after processing the map.'
)
args = parser.parse_args()

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

if args.model == 'dipole':
    model = Dipole(masked_dmap, likelihood='poisson_rms', rms_map=masked_rmsmap)
    step = False
else:
    model = Multipole(masked_dmap, ells=[0,1,2], rms_map=masked_rmsmap)
    step = True

model.prior.plot_priors()
model.run_nested_sampling(step=step)
model.corner_plot()
model.sky_direction_posterior()
