# %%
# Import packages
from dipoleska.models.dipole import Dipole
from dipoleska.models.priors import Prior
from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.modulated_map_generator import ModulatedMapGenerator
import healpy as hp
import numpy as np
import scipy as sp
from astropy.coordinates import Galactic, FK5
from astropy import units as u

# %%
# Specify parameters
x = 1.16
alpha = 0.78
cmb_velocity = 369.82e3
cmb_galactic_longitude = 264.021
cmb_galactic_latitude = 48.253
epsilon = 1e-10

# %%
# Compute derived quantities
galactic_coords = Galactic(l=cmb_galactic_longitude * u.deg, b=cmb_galactic_latitude * u.deg)
equatorial_coords = galactic_coords.transform_to(FK5(equinox='J2000'))
cmb_equatorial_ra = equatorial_coords.ra.rad
cmb_equatorial_dec = equatorial_coords.dec.rad
speed_of_light = sp.constants.c
cmb_amplitude = (2+x*(1+alpha))*cmb_velocity/speed_of_light

# %%
# Define function to run models
def run_models(MAP_FOR_INFERENCE: np.ndarray,
               ) -> tuple[Dipole, Dipole]:
    '''
    Run a fixed dipole model (M0) and a free dipole model (M1) on the
    provided map.

    :param MAP_FOR_INFERENCE: Healpy map of shape (n_pixels,).
    :return: Tuple of the two models.
    '''
    mean_count = np.mean(MAP_FOR_INFERENCE)

    ### Setup and run Model M0 (fixed dipole)
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
    # Check that priors have been defined correctly
    # model0.prior.plot_priors()
    # Run NS
    model0.run_nested_sampling()

    ### Setup and run Model M1 (free dipole)
    # Define the model
    model1 = Dipole(MAP_FOR_INFERENCE, likelihood='poisson')
    # Check that priors have been defined correctly
    # model1.prior.plot_priors()
    # Run NS
    model1.run_nested_sampling()

    return model0, model1

# %%
map_index = 1
map_setting = 'AA'

# Prepare the input map
density_map = MapLoader(map_index, map_setting).load(1)
density_map_downscaled = hp.ud_grade(density_map, nside_out=64, power=-2)

# Prepare the modulated map
modulator = ModulatedMapGenerator(
    density_map,
    dipole_amplitude=cmb_amplitude,
    dipole_longitude=cmb_galactic_longitude,
    dipole_latitude=cmb_galactic_latitude
)
dipole_map = modulator.modulated_map(scaling_factor=100)
dipole_map_downscaled = hp.ud_grade(dipole_map, nside_out=64, power=-2)

# %%

# Run models on the input map
density_map_model0, density_map_model1 = run_models(density_map_downscaled)
density_map_bayes_factor_10 = density_map_model1.log_bayesian_evidence - density_map_model0.log_bayesian_evidence
print(f"Bayes factor for the input map: {density_map_bayes_factor_10:.2f}")

# %%
density_map_model0.corner_plot(coordinates=['equatorial'])
# %%
density_map_model1.corner_plot(coordinates=['equatorial'])
# %%
# Run models on the modulated map
dipole_map_model0, dipole_map_model1 = run_models(dipole_map_downscaled)
dipole_map_bayes_factor_10 = dipole_map_model1.log_bayesian_evidence - dipole_map_model0.log_bayesian_evidence
print(f"Bayes factor for the modulated map: {dipole_map_bayes_factor_10:.2f}")

# %%
dipole_map_model0.corner_plot(coordinates=['equatorial'])

# %%
dipole_map_model1.corner_plot(coordinates=['equatorial'])

# %%
# Corner-overplot the free dipole fit for the input map and the modulated map
density_map_model1.corner_plot_double(dipole_map_model1)

# Contour-overplot the free dipole fit for the input map and the modulated map
# TODO
# %%
