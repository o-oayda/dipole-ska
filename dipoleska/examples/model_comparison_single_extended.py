# %%
### Import packages
from dipoleska.models.dipole import Dipole
from dipoleska.models.multipole import Multipole
from dipoleska.models.priors import Prior
from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.modulated_map_generator import ModulatedMapGenerator
import healpy as hp
import numpy as np
import scipy as sp
from astropy.coordinates import Galactic, FK5
from astropy import units as u
import matplotlib.pyplot as plt
import os

# Models tested:
# - M0: Monopole
# - M1: Monopole + Dipole (Free)
# - M2: Monopole + Dipole (Fixed to CMB Expectation)
# - M3: Monopole + Dipole (Free) + Quadrupole
# - M4: Monopole + Dipole (Fixed to CMB Expectation) + Quadrupole

# Sky types tested:
# - full: full sky map
# - masked: full sky map with galactic plane masked
# - southern: southern sky map (north equatorial pole masked)
# - southern_masked: southern sky map (north equatorial pole masked) with galactic plane masked

# Dipole amplitudes tested:
# - 1: CMB dipole amplitude
# - 2: CMB dipole amplitude scaled by a factor of 2

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
               OUTPUT_DIR: str,
               ) -> tuple[Dipole, Dipole]:
    '''
    Run a fixed dipole model (M0) and a free dipole model (M1) on the
    provided map.

    :param MAP_FOR_INFERENCE: Healpy map of shape (n_pixels,).
    :return: Tuple of the two models.
    '''
    mean_count = np.mean(MAP_FOR_INFERENCE)

    ### Setup and run model M0: Monopole
    # Define the model
    monopole = Multipole(MAP_FOR_INFERENCE, ells=[0])
    # Run NS
    monopole.run_nested_sampling(reactive_sampler_kwargs={'log_dir': OUTPUT_DIR + 'runs/'})

    ### Setup and run model M1: Monopole + Dipole (Free)
    # Define the model
    monopole_dipole_free = Dipole(MAP_FOR_INFERENCE, likelihood='poisson')
    # Run NS
    monopole_dipole_free.run_nested_sampling(reactive_sampler_kwargs={'log_dir': OUTPUT_DIR + 'runs/'})

    ### Setup and run model M2: Monopole + Dipole (Fixed to CMB Expectation)
    # Define the fixed prior
    prior = Prior(
        choose_prior={
            'N': ['Uniform', 0.75 * mean_count, 1.25 * mean_count],
            'D': ['Uniform', cmb_amplitude - epsilon, cmb_amplitude + epsilon],
            'phi': ['Uniform', cmb_equatorial_ra - epsilon, cmb_equatorial_ra + epsilon],
            'theta': ['Uniform', 0.5 * np.pi - cmb_equatorial_dec - epsilon, 0.5 * np.pi - cmb_equatorial_dec + epsilon],
        })
    # Define the model
    monopole_dipole_fixed = Dipole(MAP_FOR_INFERENCE, prior=prior, likelihood='poisson')
    # Run NS
    monopole_dipole_fixed.run_nested_sampling(reactive_sampler_kwargs={'log_dir': OUTPUT_DIR + 'runs/'})

    ### Setup and run model M3: Monopole + Dipole (Free) + Quadrupole
    # Define the model
    monopole_dipole_free_quadrupole = Multipole(MAP_FOR_INFERENCE, ells=[0,1,2])
    # Run NS
    monopole_dipole_free_quadrupole.run_nested_sampling(reactive_sampler_kwargs={'log_dir': OUTPUT_DIR + 'runs/'})

    ### Setup and run model M4: Monopole + Dipole (Fixed to CMB Expectation) + Quadrupole
    # Define the fixed prior
    prior = Prior(
        choose_prior={
            'M0': ['Uniform', 0.75 * mean_count, 1.25 * mean_count],
            'M1': ['Uniform', cmb_amplitude - epsilon, cmb_amplitude + epsilon],
            'M2': ['Uniform', 0., 0.4], # default for quadrupole
            'phi_l1_0': ['Uniform', cmb_equatorial_ra - epsilon, cmb_equatorial_ra + epsilon],
            'theta_l1_0': ['Uniform', 0.5 * np.pi - cmb_equatorial_dec - epsilon, 0.5 * np.pi - cmb_equatorial_dec + epsilon],
            'phi_l2_0': ['Uniform', 0., 2 * np.pi], # default for quadrupole
            'theta_l2_0': ['Polar', 0., np.pi], # default for quadrupole
            'phi_l2_1': ['Uniform', 0., 2 * np.pi], # default for quadrupole
            'theta_l1_1': ['Polar', 0., np.pi], # default for quadrupole
        })
    # Define the model
    monopole_dipole_fixed_quadrupole = Multipole(MAP_FOR_INFERENCE, prior=prior, ells=[0,1,2])
    # Run NS
    monopole_dipole_fixed_quadrupole.run_nested_sampling(reactive_sampler_kwargs={'log_dir': OUTPUT_DIR + 'runs/'})

    return monopole, monopole_dipole_free, monopole_dipole_fixed, monopole_dipole_free_quadrupole, monopole_dipole_fixed_quadrupole

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
output_path_base = f'output/ska/briggs_{briggs_weighting_str}/{configuration}/map_{map_number}/'
os.makedirs(output_path_base, exist_ok=True)

# %%
### Run block for amplitudes and sky types
dipole_amplitudes = [1,2] # in units of D/D_CMB
sky_types = ['full_sky', 'galactic_masked', 'southern_sky', 'southern_sky_galactic_masked']

for dipole_amplitude in dipole_amplitudes:
    output_path = output_path_base + f'{dipole_amplitude}xCMB/'
    os.makedirs(output_path, exist_ok=True)
    for sky_type in sky_types:
        output_path = output_path + f'{sky_type}/'
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path+'runs/', exist_ok=True)

        ### Preparation

        # Select mask
        galactic_plane_mask_radius = 10 # radius degrees +/- of galactic plane
        north_equatorial_mask_radius = 70 # radius degrees of north equatorial pole

        galactic_plane_mask_radius = galactic_plane_mask_radius if 'masked' in sky_type else 0
        north_equatorial_mask_radius = north_equatorial_mask_radius if 'southern' in sky_type else 0

        # Prepare the input map
        input_map = MapLoader(briggs_weighting, configuration).load(map_number)
        processor = MapProcessor(input_map)
        processor.change_map_resolution(nside_out=64)
        processor.mask(
            classification=['north_equatorial','galactic_plane'],
            radius=[north_equatorial_mask_radius,galactic_plane_mask_radius],
            output_frame='G'
        )
        input_map_downscaled = processor.density_map

        # Prepare the modulated map
        modulator = ModulatedMapGenerator(
            input_map,
            dipole_amplitude=cmb_amplitude*dipole_amplitude,
            dipole_longitude=np.rad2deg(cmb_equatorial_ra),
            dipole_latitude=np.rad2deg(cmb_equatorial_dec)
        )
        modulated_map = modulator.modulated_map(scaling_factor=100)
        processor2 = MapProcessor(modulated_map)
        processor2.change_map_resolution(nside_out=64)
        processor2.mask(
            classification=['north_equatorial','galactic_plane'],
            radius=[north_equatorial_mask_radius,galactic_plane_mask_radius],
            output_frame='G'
        )
        modulated_map_downscaled = processor2.density_map

        ### Run models
        # Run models on the input map
        input_monopole, input_monopole_dipole_free, input_monopole_dipole_fixed, input_monopole_dipole_free_quadrupole, input_monopole_dipole_fixed_quadrupole = run_models(input_map_downscaled, output_path)
        # Calculate the Bayes factors
        input_Z_null = input_monopole.log_bayesian_evidence # null is model 0
        input_B_10 = input_monopole_dipole_free.log_bayesian_evidence - input_Z_null # free dipole vs null
        input_B_20 = input_monopole_dipole_fixed.log_bayesian_evidence - input_Z_null # fixed dipole vs null
        input_B_30 = input_monopole_dipole_free_quadrupole.log_bayesian_evidence - input_Z_null # free dipole + quadrupole vs null
        input_B_40 = input_monopole_dipole_fixed_quadrupole.log_bayesian_evidence - input_Z_null # fixed dipole + quadrupole vs null

        # Run models on the modulated map
        modulated_monopole, modulated_monopole_dipole_free, modulated_monopole_dipole_fixed, modulated_monopole_dipole_free_quadrupole, modulated_monopole_dipole_fixed_quadrupole = run_models(modulated_map_downscaled, output_path)
        # Calculate the Bayes factors
        modulated_Z_null = modulated_monopole.log_bayesian_evidence # null is model 0
        modulated_B_10 = modulated_monopole_dipole_free.log_bayesian_evidence - modulated_Z_null # free dipole vs null
        modulated_B_20 = modulated_monopole_dipole_fixed.log_bayesian_evidence - modulated_Z_null # fixed dipole vs null
        modulated_B_30 = modulated_monopole_dipole_free_quadrupole.log_bayesian_evidence - modulated_Z_null # free dipole + quadrupole vs null
        modulated_B_40 = modulated_monopole_dipole_fixed_quadrupole.log_bayesian_evidence - modulated_Z_null # fixed dipole + quadrupole vs null

        # ### Save outputs (commented out for now - can produce outputs later)

        # # Save the Bayes factors
        # np.savetxt(output_path + f'input_bayes_factors.txt', np.array([input_B_10, input_B_20, input_B_30, input_B_40]), fmt='%.6f')
        # np.savetxt(output_path + f'modulated_bayes_factors.txt', np.array([modulated_B_10, modulated_B_20, modulated_B_30, modulated_B_40]), fmt='%.6f')

        # # Save cornerplot of M1: Monopole + Dipole (Free) for both the input and modulated maps
        # input_monopole_dipole_free.corner_plot_double(modulated_monopole_dipole_free, coordinates=['equatorial','galactic'], labels=['Input (M1)', 'Modulated (M1)'], save_path=output_path + f'M1_cornerplot.png')

        # # Save cornerplot of M3: Monopole + Dipole (Free) + Quadrupole for both the input and modulated maps
        # input_monopole_dipole_free_quadrupole.corner_plot_double(modulated_monopole_dipole_free_quadrupole, coordinates=['equatorial','galactic'], labels=['Input (M3)', 'Modulated (M3)'], save_path=output_path + f'M3_cornerplot.png')

        # # Save cornerplot of M4: Monopole + Dipole (Fixed to CMB Expectation) + Quadrupole for both the input and modulated maps
        # input_monopole_dipole_fixed_quadrupole.corner_plot_double(modulated_monopole_dipole_fixed_quadrupole, coordinates=['equatorial','galactic'], labels=['Input (M4)', 'Modulated (M4)'], save_path=output_path + f'M4_cornerplot.png')

        # # Save sky posteriors of M1: Monopole + Dipole (Free) for both the input and modulated maps
        # input_monopole_dipole_free.sky_direction_posterior(instantiate_new_axes=True, colour='cornflowerblue', label='Input (M1)')
        # modulated_monopole_dipole_free.sky_direction_posterior(instantiate_new_axes=False, colour='tomato', label='Modulated (M1)')
        # plt.savefig(output_path + f'M1_sky_direction.png', dpi=300, bbox_inches='tight')

        # # TODO: Implement functionality for sky posteriors of quadrapoles
        # # # Save sky posteriors of M3: Monopole + Dipole (Free) + Quadrupole for both the input and modulated maps
        # # input_monopole_dipole_free_quadrupole.sky_direction_posterior(instantiate_new_axes=True, colour='cornflowerblue', label='Input (M3)')
        # # modulated_monopole_dipole_free_quadrupole.sky_direction_posterior(instantiate_new_axes=False, colour='tomato', label='Modulated (M3)')
        # # plt.savefig(output_path + f'M3_sky_direction.png', dpi=300, bbox_inches='tight')

        # # # Save sky posteriors of M4: Monopole + Dipole (Fixed to CMB Expectation) + Quadrupole for both the input and modulated maps
        # # input_monopole_dipole_fixed_quadrupole.sky_direction_posterior(instantiate_new_axes=True, colour='cornflowerblue', label='Input (M4)')
        # # modulated_monopole_dipole_fixed_quadrupole.sky_direction_posterior(instantiate_new_axes=False, colour='tomato', label='Modulated (M4)')
        # # plt.savefig(output_path + f'M4_sky_direction.png', dpi=300, bbox_inches='tight')