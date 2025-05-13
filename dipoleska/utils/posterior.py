import numpy as np
from numpy.typing import NDArray
from corner import corner
import matplotlib.pyplot as plt
import os
from dipoleska.utils.physics import change_source_coordinates

class Posterior:
    def __init__(self,
            equal_weighted_samples: NDArray[np.float64] | None = None,
            run_number: int | None = None
    ) -> None:
        assert not (
            (equal_weighted_samples is None)
            and
            (run_number is None)
        ), 'Specify either equal weighted samples or a run path, not both.'
        
        self._load_samples_from_log(run_number)

    def _load_samples_from_log(self,
            run_number: int | None = None
    ) -> None:
        if run_number is None:
            pass
        else:
            self.load_path = (
                f'ultranest_logs/run{run_number}/chains/equal_weighted_post.txt'
            )
            assert os.path.exists(
                self.load_path
            ), f'Cannot find path ({self.load_path}).'
            
            self.samples = np.loadtxt(self.load_path, skiprows=1)
            self.parameter_names = np.loadtxt(
                self.load_path,
                max_rows=1,
                dtype=str
            )

    def _convert_samples(self, coordinates: list[str]) -> NDArray[np.float64]:
        '''
        Change coordinates of samples depending on user input. Only dipole
        conversions are supported at this stage.

        :param coordinates: See docstring of user-facing `corner_plot`.
        '''
        samples_for_corner = self.samples.copy()

        dipole_longitude_rad = self.samples[:, -2]
        dipole_colatitude_rad = self.samples[:, -1]

        dipole_longitude_deg = np.rad2deg(dipole_longitude_rad)
        dipole_latitude_deg =  np.rad2deg(np.pi / 2 - dipole_colatitude_rad)

        if len(coordinates) == 1:
            samples_for_corner[:, -2] = dipole_longitude_deg
            samples_for_corner[:, -1] = dipole_latitude_deg
            return samples_for_corner
        else:
            transformed_longitude, transformed_latitude = change_source_coordinates(
                dipole_longitude_deg,
                dipole_latitude_deg,
                native_coordinates=coordinates[0],
                target_coordinates=coordinates[1]
            )
            samples_for_corner[:, -2] = transformed_longitude
            samples_for_corner[:, -1] = transformed_latitude
            return samples_for_corner

    def corner_plot(self,
            coordinates: list[str] | None = None,
            **corner_kwargs
        ) -> None:
        '''
        Make corner plot for NS run.

        :param coordinates: Specify a list of coordinates to transform the angle
            indices of the corner plot. If the list has two elements, the first
            coordinate is assumed to be the native coordinares and the last
            the target coordinates. For example, specifying
            `coordinates=['equatorial', 'galactic']` transforms from equatorial
            to galactic. Specifying `coordinates=['equatorial']` would leave
            the corner in its native coordinates, but since sampling is done
            internally in spherical coordinates, it would also involve a
            conversion to longitude and latitude in degrees.
        '''
        if coordinates is not None:
            samples_for_corner = self._convert_samples(coordinates)
        else:
            samples_for_corner = self.samples
        
        corner(
            samples_for_corner,
            **{
                'labels': self.parameter_names,
                'bins': 50,
                'show_titles': True,
                'title_fmt': '.3g',
                'title_quantile': (0.025,0.5,0.975),
                'quantiles': (0.025,0.5,0.975),
                'smooth': 1,
                'smooth1d': 1,
                **corner_kwargs
            }
        )
        plt.show()

    def posterior_predictive_check(self):
        pass

    def sky_direction_posterior(self):
        pass