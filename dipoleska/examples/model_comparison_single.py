# %%
### Import packages
from dipoleska.models.dipole import Dipole
from dipoleska.models.priors import Prior
from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.modulated_map_generator import ModulatedMapGenerator
import healpy as hp
import numpy as np
import scipy as sp
from astropy.coordinates import Galactic, FK5
from astropy import units as u
import matplotlib.pyplot as plt

# %%
### Specify parameters
x = 1.16
alpha = 0.78
cmb_velocity = 369.82e3
cmb_galactic_longitude = 264.021
cmb_galactic_latitude = 48.253
epsilon = 1e-10

# %%
### Compute derived quantities
galactic_coords = Galactic(l=cmb_galactic_longitude * u.deg, b=cmb_galactic_latitude * u.deg)
equatorial_coords = galactic_coords.transform_to(FK5(equinox='J2000'))
cmb_equatorial_ra = equatorial_coords.ra.rad
cmb_equatorial_dec = equatorial_coords.dec.rad
speed_of_light = sp.constants.c
cmb_amplitude = (2+x*(1+alpha))*cmb_velocity/speed_of_light

# %%
### Define function to run models
def run_models(MAP_FOR_INFERENCE: np.ndarray,
               ) -> tuple[Dipole, Dipole]:
    '''
    Run a fixed dipole model (M0) and a free dipole model (M1) on the
    provided map.

    :param MAP_FOR_INFERENCE: Healpy map of shape (n_pixels,).
    :return: Tuple of the two models.
    '''
    mean_count = np.mean(MAP_FOR_INFERENCE)

    ### Setup and run Model 0 -- Fixed Dipole to CMB Expectation
    # Define the fixed prior
    prior = Prior(
        choose_prior={
            'N': ['Uniform', 0.75 * mean_count, 1.25 * mean_count],
            'D': ['Uniform', cmb_amplitude - epsilon, cmb_amplitude + epsilon],
            'phi': ['Uniform', cmb_equatorial_ra - epsilon, cmb_equatorial_ra + epsilon],
            'theta': ['Uniform', 0.5 * np.pi - cmb_equatorial_dec - epsilon, 0.5 * np.pi - cmb_equatorial_dec + epsilon],
        }
    )
    # Define the model
    model0 = Dipole(MAP_FOR_INFERENCE, prior=prior, likelihood='poisson')
    # Run NS
    model0.run_nested_sampling()

    ### Setup and run Model 1 -- Free Dipole
    # Define the model
    model1 = Dipole(MAP_FOR_INFERENCE, likelihood='poisson')
    # Run NS
    model1.run_nested_sampling()

    return model0, model1

# %%
### Pathing
# Select the map
briggs_weighting = 1
configuration = 'AA'
map_number = 1

# Set path for saving output data products
if briggs_weighting == -1:
    briggs_weighting_str = 'n1'
else:
    briggs_weighting_str = str(briggs_weighting)
output_path = f'output/ska/briggs_{briggs_weighting_str}/{configuration}/'

# %%
### Preparation
# Prepare the input map
density_map = MapLoader(briggs_weighting, configuration).load(map_number)
density_map_downscaled = hp.ud_grade(density_map, nside_out=64, power=-2)

# Prepare the modulated map
modulator = ModulatedMapGenerator(
    density_map,
    dipole_amplitude=cmb_amplitude,
    dipole_longitude=np.rad2deg(cmb_equatorial_ra),
    dipole_latitude=np.rad2deg(cmb_equatorial_dec)
)
dipole_map = modulator.modulated_map(scaling_factor=100)
dipole_map_downscaled = hp.ud_grade(dipole_map, nside_out=64, power=-2)

# %%
### Run models
# Run models on the input map
density_map_model0, density_map_model1 = run_models(density_map_downscaled)
# Calculate the Bayes factor: BF = Z_free_dipole / Z_fixed_dipole
density_map_bayes_factor_10 = density_map_model1.log_bayesian_evidence - density_map_model0.log_bayesian_evidence

# Run models on the modulated map
dipole_map_model0, dipole_map_model1 = run_models(dipole_map_downscaled)
# Calculate the Bayes factor: BF = Z_free_dipole / Z_fixed_dipole
dipole_map_bayes_factor_10 = dipole_map_model1.log_bayesian_evidence - dipole_map_model0.log_bayesian_evidence

# %%
### Save outputs
# Save the Bayes factors
np.savetxt(output_path + f'map_{map_number}_bayes_factors.txt', np.array([density_map_bayes_factor_10, dipole_map_bayes_factor_10]))
# Plot and save the free dipole posteriors for both the input and modulated maps (cornerplot)
density_map_model1.corner_plot_double(dipole_map_model1, coordinates=['equatorial','galactic'], labels=['Input Map', 'Modulated Map'], save_path=output_path + f'map_{map_number}_cornerplot.png')

# Plot and save the free dipole posteriors for both the input and modulated maps (mollview)
dipole_map_model1.sky_direction_posterior(instantiate_new_axes=True, colour='tomato', label='Modulated Map')
density_map_model1.sky_direction_posterior(instantiate_new_axes=False, colour='cornflowerblue', label='Input Map')
plt.savefig(output_path + f'map_{map_number}_sky_direction_posterior.png', dpi=300, bbox_inches='tight')
# %%
