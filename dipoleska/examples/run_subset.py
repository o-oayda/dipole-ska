from dipoleska.models.dipole import Dipole
from dipoleska.models.multipole import Multipole
from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.modulated_map_generator import ModulatedMapGenerator
from dipoleska.utils.physics import compute_ellis_baldwin_amplitude
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.plotting import MapPlotter
from dipoleska.models.priors import Prior
from numpy.typing import NDArray
from dipoleska.utils.constants import (
    CMB_RA, CMB_DEC, CMB_BETA, CMB_PHI_EQ, CMB_THETA_EQ, SKA_X, SKA_ALPHA
)
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np

def remove_previous_runs(log_dir: str) -> str | None:
    if os.path.exists(log_dir):
        for entry in os.listdir(log_dir):
            entry_path = os.path.join(log_dir, entry)
            if (
                os.path.isdir(entry_path)
                and entry.startswith('run')
                and entry[3:].isdigit()
            ):
                confirm = input(
                    f"Remove and rerun '{entry_path}'? [Y/n]: "
                ).strip()
                if confirm == 'Y':
                    shutil.rmtree(entry_path)
                    return 'rerun'
                else:
                    print('Skipping model fitting...')
                    return 'skip'

def run_monopole(density_map: NDArray[np.float64], run_dir: str) -> None:
    model = Multipole(density_map, ells=[0])
    log_dir = f'{run_dir}/monopole/'
    if remove_previous_runs(log_dir) == 'skip':
        return
    model.run_nested_sampling(
        reactive_sampler_kwargs={'log_dir': log_dir}
    )

def run_dipole(density_map: NDArray[np.float64], run_dir: 'str') -> None:
    model = Dipole(density_map, likelihood='poisson')
    log_dir = f'{run_dir}/dipole/'
    if remove_previous_runs(log_dir) == 'skip':
        return
    model.run_nested_sampling(
        reactive_sampler_kwargs={'log_dir': log_dir}
    )

def run_kinematic_dipole(
        density_map: NDArray[np.float64],
        run_dir: str,
        cmb_amplitude: float | np.floating
    ) -> None:
    mean_count = np.nanmean(density_map)
    prior = Prior(
        choose_prior={
            'N': ['Uniform', 0.75 * mean_count, 1.25 * mean_count],
            'D': ['Uniform', cmb_amplitude - 1e-8, cmb_amplitude + 1e-8],
            'phi': ['Uniform', CMB_PHI_EQ - 1e-8, CMB_PHI_EQ + 1e-8],
            'theta': ['Uniform', CMB_THETA_EQ - 1e-8, CMB_THETA_EQ + 1e-8],
        }
    )
    model = Dipole(density_map, prior=prior)
    log_dir = f'{run_dir}/kinematic_dipole/'
    if remove_previous_runs(log_dir) == 'skip':
        return
    model.run_nested_sampling(
        reactive_sampler_kwargs={'log_dir': log_dir}
    )

def run_dipole_quadrupole(density_map: NDArray[np.float64], run_dir: str) -> None:
    model = Multipole(density_map, ells=[0, 1, 2])
    log_dir = f'{run_dir}/dipole_quadrupole/'
    if remove_previous_runs(log_dir) == 'skip':
        return
    model.run_nested_sampling(
        step=True,
        reactive_sampler_kwargs={'log_dir': log_dir}
    )

def run_kinematic_dipole_quadrupole(
        density_map: NDArray[np.float64],
        run_dir: str,
        cmb_amplitude: float | np.floating
    ) -> None:
    mean_count = np.nanmean(density_map)
    log_dir = f'{run_dir}/kinematic_dipole_quadrupole/'
    if remove_previous_runs(log_dir) == 'skip':
        return
    # TODO: desperately need a check for mis-specified labels, can cause bad
    # unclear errors down the line
    prior = Prior(
        choose_prior={
            'M0': ['Uniform', 0.75 * mean_count, 1.25 * mean_count],
            'M1': ['Uniform', cmb_amplitude - 1e-8, cmb_amplitude + 1e-8],
            'M2': ['Uniform', 0., 0.4],
            'phi_l1_0': ['Uniform', CMB_PHI_EQ - 1e-8, CMB_PHI_EQ + 1e-8],
            'theta_l1_0': ['Uniform', CMB_THETA_EQ - 1e-8, CMB_THETA_EQ + 1e-8],
            'phi_l2_0': ['Uniform', 0., 2 * np.pi],
            'theta_l2_0': ['Polar', 0., np.pi],
            'phi_l2_1': ['Uniform', 0., 2 * np.pi],
            'theta_l2_1': ['Polar', 0., np.pi],
        }
    )
    model = Multipole(density_map, ells=[0,1,2], prior=prior)
    model.run_nested_sampling(
        step=True,
        reactive_sampler_kwargs={'log_dir': log_dir}
    )

def add_kinematic_dipole(
        density_map: NDArray[np.int_],
        amplitude: float | np.floating
    ) -> NDArray[np.int_]:
    modulator = ModulatedMapGenerator(
        density_map,
        dipole_amplitude=amplitude,
        dipole_longitude=CMB_RA,
        dipole_latitude=CMB_DEC
    )
    modulated_density_map = modulator.modulated_map()
    return modulated_density_map

def full_sky_mask(processor: MapProcessor) -> MapProcessor:
    return processor

def northern_sky_mask(processor: MapProcessor) -> MapProcessor:
    processor.mask(
        classification=['north_equatorial'],
        radius=[70],
        output_frame='C'
    )
    return processor

def northern_and_galactic_mask(processor: MapProcessor) -> MapProcessor:
    processor.mask(
        classification=['north_equatorial', 'galactic_plane'],
        radius=[70, 10],
        output_frame='C'
    )
    return processor

def main():
    BRIGGS = 1
    CONFIG = 'AA'
    MAPS_TO_TEST = [1]
    MASKS = [full_sky_mask, northern_sky_mask, northern_and_galactic_mask]
    MASK_NAMES = ['full', 'northern', 'northern_galactic']
    BASE_DIR = f'output/ska/briggs_{BRIGGS}/{CONFIG}'
    AMP_MULTIPLIERS = [1, 2]
    
    cmb_amplitude = compute_ellis_baldwin_amplitude(CMB_BETA, SKA_X, SKA_ALPHA)
    loader = MapLoader(briggs_weighting=1, configuration='AA')

    for multiplier in AMP_MULTIPLIERS:
        for map_number in MAPS_TO_TEST:
            density_map = loader.load(map_number)
            modulated_density_map = add_kinematic_dipole(
                density_map=density_map,
                amplitude=multiplier * cmb_amplitude
            )
            processor = MapProcessor(modulated_density_map)
            processor.change_map_resolution(nside_out=64)
            
            for mask_name, mask in zip(MASK_NAMES, MASKS):
                processor = mask(processor=processor)
                map_to_test = processor.density_map

                # plotter = MapPlotter(map_to_test)
                # plotter.plot_density_map(); plt.show()
                
                run_dir = f'{BASE_DIR}/mult_{multiplier}/map_{map_number}/{mask_name}'
                run_monopole(map_to_test, run_dir)
                run_dipole(map_to_test, run_dir)
                run_kinematic_dipole(map_to_test, run_dir, cmb_amplitude)
                run_dipole_quadrupole(map_to_test, run_dir)
                run_kinematic_dipole_quadrupole(map_to_test, run_dir, cmb_amplitude)

                processor.reset_mask()

if __name__ == '__main__':
    main()