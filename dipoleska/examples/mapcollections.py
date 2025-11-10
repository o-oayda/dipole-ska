'''
Example script for loading in the new map collections (10.11.25).

Notes:
- snr=10, S_min=1e-4, z_min=0.5, gal_cut=10: ring of fire (?) kind of effect,
    strong declination-dependent systematic
'''
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapCollectionLoader
from dipoleska.utils.plotting import MapPlotter
from dipoleska.models.multipole import Multipole
import matplotlib.pyplot as plt
from dipoleska.models.dipole import Dipole
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--no-plots',
    action='store_true',
    help='Disable all plotting so the script can run in headless environments.'
)
parser.add_argument(
    '--model',
    choices=['dipole', 'multipole'],
    default='multipole',
    help='Choose which model to fit after processing the map.'
)
args = parser.parse_args()
plot_enabled = not args.no_plots

loader = MapCollectionLoader(
    snr_cut=10, 
    lower_flux_limit='1e-4',
    lower_z_limit='0.5',
    gal_cut=10,
    map_types=['all']
)
data = loader.map_collections

dmap = data['counts']
processor = MapProcessor(dmap)
processor.mask(output_frame='C', load_from_file='gal10_ps')
masked_dmap = processor.density_map

if plot_enabled:
    plotter = MapPlotter(masked_dmap)
    plotter.plot_density_map(projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']})
    plotter.plot_smooth_map(projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']})
    plt.show()

if args.model == 'dipole':
    model = Dipole(masked_dmap, likelihood='poisson')
else:
    model = Multipole(masked_dmap, ells=[1, 2])

model.run_nested_sampling()

if plot_enabled:
    model.corner_plot(backend='getdist', coordinates=['equatorial', 'galactic'])
    model.sky_direction_posterior(coordinates=['equatorial', 'galactic'])
    plt.show()
