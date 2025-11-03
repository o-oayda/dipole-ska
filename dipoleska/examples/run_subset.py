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
from typing import Literal, Callable, cast

class SKASim:
    '''
    Class for iterating over SKA mocks and fitting various models.
    '''
    def __init__(self,
            briggs_weighting: list[Literal[-1, 0, 1]],
            config: list[Literal['AA', 'AA4']],
            maps: list[int] | Literal['all'],
            masks: list[Literal['full', 'northern', 'northern_galactic']],
            amplitude_multipliers: list[Literal[1, 2]],
            models: list[
                Literal[
                    'monopole',
                    'dipole',
                    'kinematic_dipole',
                    'dipole_quadrupole',
                    'kinematic_dipole_quadrupole'
                ]
            ]
    ) -> None:
        self.briggs = briggs_weighting
        self.configs = config
        self.maps = maps
        self.masks = masks
        self.amplitudes = amplitude_multipliers
        self.models = models
        self.cmb_amplitude = compute_ellis_baldwin_amplitude(
            CMB_BETA, SKA_X, SKA_ALPHA
        )
        self.base_dir = f'output/ska'
    
    def run_simulations(self) -> None:

        for briggs_num in self.briggs:
            for config in self.configs:
                loader = MapLoader(
                    briggs_weighting=briggs_num,
                    configuration=config
                )
                
                if self.maps == 'all':
                    maps = self._get_all_map_nums(briggs_num, config)
                else:
                    maps = cast(list[int], self.maps) # to make sure the typechecker knows this is a list of ints

                for map_num in maps:
                    density_map = loader.load(map_num)

                    for multiplier in self.amplitudes:
                        modulated_density_map = self._add_kinematic_dipole(
                            density_map=density_map,
                            amplitude=multiplier * self.cmb_amplitude
                        )

                        processor = MapProcessor(modulated_density_map)
                        processor.change_map_resolution(nside_out=64)
                        map_dir = self._make_run_directory(
                            briggs_num, config, multiplier, map_num, None
                        )
                        
                        self._save_map(map_dir, processor.density_map)

                        for mask in self.masks:
                            mask_function = self._mask_name_to_function(mask)
                            mask_function(processor)
                            self.map_to_test = processor.density_map
                            self.run_dir = self._make_run_directory(
                                briggs_num, config, multiplier, map_num, mask
                            )
                            
                            for model in self.models:
                                model_func = self._model_name_to_function(model)
                                model_args = self._model_name_to_args(model)
                                model_func(*model_args)
                            
                            processor.reset_mask()
    
    def _model_name_to_args(self,
            model_name: Literal[
                'monopole',
                'dipole',
                'kinematic_dipole',
                'dipole_quadrupole',
                'kinematic_dipole_quadrupole'
            ]
    ) -> tuple:
        if 'kinematic' in model_name:
            return self.map_to_test, self.run_dir, self.cmb_amplitude
        else:
            return self.map_to_test, self.run_dir

    def _get_all_map_nums(self,
            briggs_num: Literal[-1, 0, 1],
            config: str
    ) -> list[int]:
        '''
        Turn `map_i.fits` --> `i` for all map numbers `i` in the directory. 
        '''
        return [
            int(f.split('_')[1].split('.')[0])
            for f in os.listdir(f'output/ska/briggs_{briggs_num}/{config}/')
            if f.startswith('map_') and f.endswith('.fits')
        ]

    def _save_map(self, map_dir: str, density_map: NDArray[np.float64]) -> None:
        os.makedirs(map_dir, exist_ok=True)
        np.save(
            os.path.join(map_dir, "modulated_density_map.npy"),
            density_map
        )

    def _make_run_directory(self,
            briggs: Literal[-1, 0, 1],
            config: Literal['AA', 'AA4'],
            multiplier: Literal[1, 2],
            map_number: int,
            mask_name: Literal['full', 'northern', 'northern_galactic'] | None
    ) -> str:
        if briggs == -1:
            self.briggs_weighting = 'n1'
        else:
            self.briggs_weighting = str(briggs)
        
        run_dir = (
            f'{self.base_dir}/briggs_{briggs}/{config}/mult_{multiplier}/'
            f'map_{map_number}'
        )
        if mask_name is not None:
            return f'{run_dir}/{mask_name}'
        else:
            return run_dir
        
    def _add_kinematic_dipole(self,
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

    def _full_sky_mask(self, processor: MapProcessor) -> MapProcessor:
        return processor

    def _northern_sky_mask(self, processor: MapProcessor) -> MapProcessor:
        processor.mask(
            classification=['north_equatorial'],
            radius=[70],
            output_frame='C'
        )
        return processor

    def _northern_and_galactic_mask(self, processor: MapProcessor) -> MapProcessor:
        processor.mask(
            classification=['north_equatorial', 'galactic_plane'],
            radius=[70, 10],
            output_frame='C'
        )
        return processor
    
    def _mask_name_to_function(self,
            mask_name: Literal['full', 'northern', 'northern_galactic']
    ) -> Callable:
        name_to_func = {
            'full': self._full_sky_mask,
            'northern': self._northern_sky_mask,
            'northern_galactic': self._northern_and_galactic_mask
        }
        return name_to_func[mask_name]
    
    def _model_name_to_function(self,
            model_name: Literal[
                'monopole',
                'dipole',
                'kinematic_dipole',
                'dipole_quadrupole',
                'kinematic_dipole_quadrupole'
            ]
    ) -> Callable:
        return getattr(self, f'_run_{model_name}')

    def _run_monopole(self, density_map: NDArray[np.float64], run_dir: str) -> None:
        model = Multipole(density_map, ells=[0])
        log_dir = f'{run_dir}/monopole/'
        if self._remove_previous_runs(log_dir) == 'skip':
            return
        model.run_nested_sampling(
            reactive_sampler_kwargs={'log_dir': log_dir}
        )

    def _run_dipole(self, density_map: NDArray[np.float64], run_dir: 'str') -> None:
        model = Dipole(density_map, likelihood='poisson')
        log_dir = f'{run_dir}/dipole/'
        if self._remove_previous_runs(log_dir) == 'skip':
            return
        model.run_nested_sampling(
            reactive_sampler_kwargs={'log_dir': log_dir}
        )

    def _run_kinematic_dipole(self,
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
        model = Dipole(density_map, likelihood='poisson', prior=prior)
        log_dir = f'{run_dir}/kinematic_dipole/'
        if self._remove_previous_runs(log_dir) == 'skip':
            return
        model.run_nested_sampling(
            reactive_sampler_kwargs={'log_dir': log_dir}
        )

    def _run_dipole_quadrupole(self, density_map: NDArray[np.float64], run_dir: str) -> None:
        model = Multipole(density_map, ells=[0, 1, 2])
        log_dir = f'{run_dir}/dipole_quadrupole/'
        if self._remove_previous_runs(log_dir) == 'skip':
            return
        model.run_nested_sampling(
            step=True,
            reactive_sampler_kwargs={'log_dir': log_dir}
        )

    def _run_kinematic_dipole_quadrupole(self,
            density_map: NDArray[np.float64],
            run_dir: str,
            cmb_amplitude: float | np.floating
        ) -> None:
        mean_count = np.nanmean(density_map)
        log_dir = f'{run_dir}/kinematic_dipole_quadrupole/'
        if self._remove_previous_runs(log_dir) == 'skip':
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

    def _remove_previous_runs(self, log_dir: str) -> str | None:
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

def main():
    sim = SKASim(
        briggs_weighting=[1],
        config=['AA'],
        maps=[1],
        masks=['full', 'northern', 'northern_galactic'],
        amplitude_multipliers=[1, 2],
        models=['monopole', 'dipole', 'kinematic_dipole']
    )
    sim.run_simulations()

if __name__ == '__main__':
    main()