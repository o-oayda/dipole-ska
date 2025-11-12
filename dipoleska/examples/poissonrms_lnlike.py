# This script demonstrates the use of the Dipole model with a Poisson likelihood 
# that incorporates RMS noise in the data. The likelihood function is taken from
# Wagenvald et al. (2023). This likelihood scales the number density by a
# power-law function of the RMS noise map. So, for each pixel, the rate parameter
# is given by:
#
#     λ_i = M * (rms_i / rms_ref)^(-x_rms) * (1 + d̂ · n̂_i)
#
# where M is a proxy for mean density, rms_i is the RMS noise in pixel i,
# rms_ref is the reference RMS value (depends of the telescope's actual beam
# size), x_rms is the power-law index that describes how the number density scales
# with RMS noise, d̂ is the dipole direction, and n̂_i is the unit vector pointing
# to pixel i. Note that rms_ref can be chosen as the median RMS noise across the 
# map: then M corresponds to the monopole density - we implement this in our code.
# A more general implementation involves using the actual rms_ref, which will be 
# available once SKA commences observations.

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
