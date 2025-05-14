import numpy as np
from numpy.typing import NDArray
from corner import corner
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os
from dipoleska.utils.physics import (
    change_source_coordinates, spherical_to_degrees
)
import healpy as hp
from scipy.interpolate import interp1d
from dipoleska.utils.math import sigma_to_prob2D

class Posterior:
    def __init__(self,
            equal_weighted_samples: NDArray[np.float64] | int
    ) -> None:
        '''
        :param equal_weighted_samples: Can specify either an array of samples,
            or an integer referring to a run number in ultranest_logs/, in which
            case the samples from the run are automatically loaded.
        '''
        if type(equal_weighted_samples) is int:
            run_number = equal_weighted_samples
            self._load_samples_from_log(run_number)
        
        elif type(equal_weighted_samples) is np.ndarray:
            self.samples = equal_weighted_samples
        
        else:
            raise Exception('Pass either array of samples or an integer (run number).')

    def _load_samples_from_log(self,
            run_number: int
    ) -> None:
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

    def _convert_samples(self,
            coordinates: list[str] | None
        ) -> NDArray[np.float64]:
        '''
        Change coordinates of samples depending on user input. Only dipole
        conversions are supported at this stage.

        :param coordinates: See docstring of user-facing `corner_plot`. If the
            coordinates are None, the function will just perform a spherical
            (radians, colatitude) to spherical (degrees, latitude) conversion.
        '''
        samples_for_corner = self.samples.copy()

        dipole_longitude_rad = self.samples[:, -2]
        dipole_colatitude_rad = self.samples[:, -1]

        dipole_longitude_deg, dipole_latitude_deg = spherical_to_degrees(
            dipole_longitude_rad, dipole_colatitude_rad
        )

        if (coordinates is None) or ((len(coordinates) == 1)):
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

    def sky_direction_posterior(self,
            coordinates: list[str] | None = None,
            colour: str = 'tomato',
            smooth: None | float = 0.05,
            contour_levels: list[float] = [0.5, 1., 1.5, 2.],
            xsize: int = 500,
            nside: int = 256,
            rasterize_probability_mesh: bool = False,
        ) -> None:
        '''
        :param colour: Specify the matplotlib colour for the sky direction.
        :param smooth: The sigma of the Gaussian kernel used to smooth the
            healpy sample map using healpy's smoothing function.
        :param contour_levels: The significance levels (in units of sigma)
            defining how the contours are drawn.
        :param xsize: This specifies the resolution of the projection of the
            healpy map into matplotlib coordinates.
        :param nside: The resolution of the healpy map into which the samples
            are binned.
        :param rasterize_probability_mesh: Specify whether or not to rasterize
            the pcolormesh representing probability, which tends to greatly
            increase the file size if not rasterized for high xsize.
        '''
        # ensure angle samples are in degrees of longitude and latitude
        full_samples_for_sky = self._convert_samples(coordinates)

        dipole_longitude_deg = full_samples_for_sky[:, -2]
        dipole_latitude_deg = full_samples_for_sky[:, -1]

        probability_map = self._samples_to_healpy_map(
            dipole_longitude_deg,
            dipole_latitude_deg,
            lonlat=True,
            smooth=smooth,
            nside=nside
        )

        phi, theta, projected_map = hp.projview(
            probability_map,
            return_only_data=True,
            xsize=xsize
        )

        # NOTE: the projected map will have more bins therefore will not sum to
        # one; it also has -infs; remove these and renormalise
        projected_map[projected_map == -np.inf] = 0
        projected_map /= np.sum(projected_map)
        probability_contours = self._compute_2D_contours(
            projected_map,
            contour_levels
        )

        # draw sky plots
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        transparent_cmap = self._make_transparent_colour_map(colour)
        
        hp.projview(
            np.zeros(12),
            **{
                'longitude_grid_spacing': 30,
                'color': 'white',
                'graticule': True,
                'graticule_labels': True,
                'cbar': False
            }
        )
        plt.contourf(
            phi_grid,
            theta_grid,
            projected_map,
            levels=probability_contours,
            cmap=transparent_cmap,
            zorder=1,
            extend='both'
        )
        plt.contour(
            phi_grid,
            theta_grid,
            projected_map,
            levels=probability_contours,
            colors=[matplotlib.colors.to_rgba(colour)], # type: ignore
            zorder=1,
            extend='both'
        )
        plt.pcolormesh(
            phi_grid,
            theta_grid,
            projected_map,
            cmap=transparent_cmap,
            rasterized=rasterize_probability_mesh,
        )

        plt.show()

    def _make_transparent_colour_map(self,
                colour: str
        ) -> LinearSegmentedColormap:
        '''
        Make colour maps going from transparent to the desired colour.

        :param: Desired matplotlib colour.
        '''
        white_rgba = matplotlib.colors.colorConverter.to_rgba(
            'white', alpha=0 # type: ignore
        )
        chosen_rgba = matplotlib.colors.to_rgba(colour, alpha=0.4) # type: ignore
        cmap_transparent = matplotlib.colors.LinearSegmentedColormap.from_list(
            'rb_cmap',[white_rgba, chosen_rgba], 512
        )
        return cmap_transparent

    def _samples_to_healpy_map(self,
            phi: NDArray[np.float64],
            theta: NDArray[np.float64],
            lonlat: bool = False,
            nside: int = 64,
            smooth: None | float = None
        ) -> NDArray[np.float64]:
        '''
        Turn equal-weighted numerical samples in phi-theta space to a healpy map,
        in the native coords of phi theta, defining the probability of a sample
        (phi_i, theta_i) lying in a given pixel. In other words, do a 2D
        histogram and project onto a healpy map. 

        :param phi: Vector of phi samples.
        :param theta: Vector of theta samples.
        :param lonlat: If True, phi ~ [0, 360] and theta ~ [-90, 90]; else,
            phi ~ [0, 2pi] and theta ~ [0, pi].
        :param nside: Nside (resolution) of binning of nested samples, i.e. the
            resolution of the posterior probability map.
        :return: Healpy map of the total probability of sources lying in a given
            pixel.  
        '''
        # NOTE: kwargs theta and phi flip where lonlat=True... thanks healpy
        if lonlat:
            sample_pixel_indices = hp.ang2pix(
                nside=nside, theta=phi, phi=theta, lonlat=lonlat
            )
        else:
            sample_pixel_indices = hp.ang2pix(
                nside=nside, theta=theta, phi=phi
            )
        sample_count_map = np.bincount(
            sample_pixel_indices, minlength=hp.nside2npix(nside)
        )
        
        # convert count to probability
        map_total = np.sum(sample_count_map)
        sample_pdensity_map = sample_count_map / map_total
        
        if smooth is not None:
            # NOTE: healpy's smooth function (in samples_to_hpmap) works in
            # spherical harmonic space, and produces a small number of very small
            # negative values; the sum is also not preserved.
            # correct by replacing negative values with 0 and renormalise.
            smooth_map = hp.sphtfunc.smoothing(sample_pdensity_map, sigma=smooth)
            smooth_map[smooth_map < 0] = 0
            smooth_map /= np.sum(smooth_map)
            return smooth_map
        else:
            return sample_pdensity_map
    
    def _compute_2D_contours(self,
        P_xy: NDArray[np.float64],
        contour_levels: list[float]
    ) -> NDArray[np.float64]:
        '''
        Compute contour heights corresponding to sigma levels of probability
        density by creating a mapping (interpolation function) from the
        enclosed probability to some arbitrary level of probability density.
        Adapted from this link:
        https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution

        :param P_xy: normalised 2D probability (not density)
        :param contour_levels: pass list of sigmas at which to draw the contours
        :return: Vector of probabilities corresponding to heights at which to
            draw the contours (pass to e.g. plt.contour with levels=).
        '''
        levels_for_interpolation = np.linspace(0, P_xy.max(), 1000)
        mask = (P_xy >= levels_for_interpolation[:, None, None])
        enclosed_probability = (mask * P_xy).sum(axis=(1,2))
        
        # interpolate between enclosed prob and probability level
        enclosed_prob_to_probability_level = interp1d(
            enclosed_probability,
            levels_for_interpolation
        )
        probability_contours = np.flip(
            enclosed_prob_to_probability_level(
                sigma_to_prob2D(contour_levels)
            )
        )
        return probability_contours