import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os
import re

from ultranest.utils import resample_equal
from dipoleska.utils.physics import (
    change_source_coordinates, spherical_to_degrees
)
import healpy as hp
from scipy.interpolate import interp1d
from dipoleska.utils.math import sigma_to_prob2D
from typing import Self, Callable, Any, Sequence
from abc import abstractmethod
from matplotlib.patches import Patch
from typing import Literal
import json
from corner import corner
from getdist import plots
from getdist.mcsamples import MCSamples
from dipoleska.style import paperplot as paperplot_style
from dipoleska.utils.plotting import (
    matplotlib_latex, _parameter_latex_label, _sanitise_parameter_name,
    ANGLE_LABEL_OVERRIDES, apply_angle_label_override
)
from pathlib import Path

SUPPORTED_ANGLE_COORDINATES: set[str] = {'galactic', 'equatorial', 'ecliptic'}
_MULTIPOLE_ANGLE_PATTERN = re.compile(r'^(phi|theta)_(l\d+_\d+)$')


class PosteriorMixin:
    '''
    Mixin class for adding functionality to the Posterior class as well as
    model classes like the Dipole class.
    '''
    @property
    @abstractmethod
    def samples(self) -> NDArray[np.float64]:
        '''
        Grab the equal-weighted posterior samples computed from an ultranest NS run.
        '''
        raise NotImplementedError('Subclasses must define equal-weighted samples.')
    
    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        raise NotImplementedError('Subclasses must define names for each parameter.')

    @abstractmethod
    def model(self, Theta: NDArray[np.float64]) -> Any:
        raise NotImplementedError('Subclasses must define a model method.')

    @property
    def weighted_samples(self) -> NDArray[np.float64]:
        '''
        Grab the importance-weighted posterior samples computed from an ultranest NS run,
        defaulting back to the equal-weighted samples if there are no weighted samples.
        '''
        weighted = getattr(self, '_weighted_samples', None)
        if weighted is None:
            return self.samples
        return weighted

    @property
    def weights(self) -> NDArray[np.float64] | None:
        return getattr(self, '_weights', None)

    @staticmethod
    def _normalise_coordinates_argument(
            coordinates: list[str] | tuple[str, ...] | None
        ) -> list[str] | None:
        '''
        Ensure the coordinates argument is well-formed before attempting any
        transformations. Accept exactly one or two coordinate frame names drawn
        from SUPPORTED_ANGLE_COORDINATES (case-insensitive).
        '''
        if coordinates is None:
            return None
        if not isinstance(coordinates, (list, tuple)):
            raise TypeError(
                'coordinates must be a list or tuple of coordinate frame names'
            )
        if len(coordinates) == 0:
            raise ValueError('coordinates must contain at least one frame name')
        if len(coordinates) > 2:
            raise ValueError('coordinates accepts at most two frame names')

        normalised: list[str] = []
        for idx, coord in enumerate(coordinates):
            if not isinstance(coord, str):
                raise TypeError(
                    f'Coordinate entry {idx} must be a string, '
                    f'got {type(coord).__name__}'
                )
            stripped = coord.strip()
            if not stripped:
                raise ValueError(f'Coordinate entry {idx} cannot be empty')
            lowered = stripped.lower()
            if lowered not in SUPPORTED_ANGLE_COORDINATES:
                raise ValueError(
                    f'Unsupported coordinate frame "{coord}". '
                    f'Expected one of {sorted(SUPPORTED_ANGLE_COORDINATES)}.'
                )
            normalised.append(lowered)
        return normalised

    def _angle_parameter_pairs(self) -> list[tuple[int, int]]:
        '''
        Identify (phi, theta) index pairs within the parameter list. Supports both
        the simple dipole parameter names ('phi', 'theta') and multipole-style
        names that look like `phi_lX_Y` / `theta_lX_Y`. This makes sure we
        keep the angular coords of each unit vector together when e.g. rotating.
        '''
        index_by_name = {
            name: idx for idx, name in enumerate(self.parameter_names)
        }
        pairs: list[tuple[int, int]] = []

        # dipole models expose bare "phi"/"theta" entries; keep them paired
        if ('phi' in index_by_name) and ('theta' in index_by_name):
            pairs.append((index_by_name['phi'], index_by_name['theta']))

        # multipole models label angles with a suffix, so match each phi suffix
        # with its theta counterpart so rotations update both columns in sync
        for name, idx in index_by_name.items():
            match = _MULTIPOLE_ANGLE_PATTERN.match(name)
            if not match:
                continue
            angle_kind, suffix = match.groups()
            if angle_kind != 'phi':
                continue
            theta_name = f'theta_{suffix}'
            theta_idx = index_by_name.get(theta_name)
            if theta_idx is None:
                continue
            pairs.append((idx, theta_idx))

        return pairs

    def _convert_samples(self,
            samples: NDArray[np.float64],
            coordinates: list[str] | tuple[str, ...] | None
        ) -> NDArray[np.float64]:
        '''
        Change coordinates of samples depending on user input for every angular
        parameter pair (dipole or multipole). We read the parameter names of the
        samples directly and regexp to infer which are longitude and latitude,
        and properly pair each longitude with the corresponding latitude if we
        are dealing with many multipole unit vectors.

        :param coordinates: See docstring of user-facing `corner_plot`. If the
            coordinates are None or a single-length list (containing one coord system),
            the function will just perform a spherical (radians, colatitude) to 
            spherical (degrees, latitude) conversion.
            Otherwise, we do this then rotate from the native coord system to
            the target system, where the native system is the first element in 
            coordinates and the target system is the second element.

        :return: Parameter samples chain rotated transformed to the target
            coordinate system.
        '''
        validated_coordinates = self._normalise_coordinates_argument(coordinates)

        samples_array = np.asarray(samples, dtype=np.float64)
        samples_for_corner = samples_array.copy()

        angle_pairs = self._angle_parameter_pairs()
        if not angle_pairs:
            return samples_for_corner

        def _transform_angles(
                longitude_deg: NDArray[np.float64],
                latitude_deg: NDArray[np.float64]
            ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            if (validated_coordinates is None) or (len(validated_coordinates) == 1):
                return longitude_deg, latitude_deg
            transformed_longitude, transformed_latitude = change_source_coordinates(
                longitude_deg,
                latitude_deg,
                native_coordinates=validated_coordinates[0],
                target_coordinates=validated_coordinates[1]
            )
            return transformed_longitude, transformed_latitude

        for phi_idx, theta_idx in angle_pairs:
            phi_rad = samples_array[:, phi_idx]
            theta_colat_rad = samples_array[:, theta_idx]
            longitude_deg, latitude_deg = spherical_to_degrees(
                phi_rad,
                theta_colat_rad
            )
            transformed_longitude, transformed_latitude = _transform_angles(
                longitude_deg,
                latitude_deg
            )
            samples_for_corner[:, phi_idx] = transformed_longitude
            samples_for_corner[:, theta_idx] = transformed_latitude

        return samples_for_corner

    def corner_plot(self,
            coordinates: list[str] | None = None,
            save_path: str | None = None,
            backend: Literal['corner', 'getdist'] = 'corner',
            **kwargs
        ) -> None:
        '''
        Make corner plot for NS run using getdist or corner.

        ## Features
        - If using getdist, we automatically set parameters with 'phi' in their
        name to periodic --- this makes sure the direction marginals wrap
        nicely at the boundaries.

        :param coordinates: Specify a list of coordinates to transform the angle
            indices of the corner plot. If the list has two elements, the first
            coordinate is assumed to be the native coordinares and the last
            the target coordinates. For example, specifying
            `coordinates=['equatorial', 'galactic']` transforms from equatorial
            to galactic. Specifying `coordinates=['equatorial']` would leave
            the corner in its native coordinates, but since sampling is done
            internally in spherical coordinates, it would also involve a
            conversion to longitude and latitude in degrees.
        :param save_path: Specify a path to save the corner plot. If None, the
            plot will not be saved. The default is None.
        :param backend: Choose whether to use the `corner` library for the
            corner plot or the `getdist` library. Note that if using getdist,
            we use the weighted samples and not the equal weighted samples,
            since the process of boostrap resampling seems to mess with the 2D
            marginals that getdist draws.
        '''
        if backend == 'corner':
            base_samples = np.asarray(self.samples, dtype=np.float64)
        else:
            base_samples = np.asarray(self.weighted_samples, dtype=np.float64)

        normalised_coordinates = self._normalise_coordinates_argument(coordinates)

        if normalised_coordinates is not None:
            samples_for_corner = self._convert_samples(
                base_samples,
                normalised_coordinates
            )
        else:
            samples_for_corner = base_samples.copy()

        self.samples_for_corner = samples_for_corner

        # determine if we need to override angular labels after a rotation
        target_coordinate: str | None = None
        if normalised_coordinates:
            target_coordinate = normalised_coordinates[-1]

        def _maybe_override_angle_label(
                name: str,
                base_label: str
            ) -> str:
            if target_coordinate is None:
                return base_label
            overrides = ANGLE_LABEL_OVERRIDES.get(target_coordinate)
            if not overrides:
                return base_label

            def _apply_override(label: str, symbol_key: str) -> str:
                return apply_angle_label_override(label, symbol_key, overrides)

            if name == 'phi':
                return _apply_override(base_label, 'phi')
            if name == 'theta':
                return _apply_override(base_label, 'theta')

            match = _MULTIPOLE_ANGLE_PATTERN.match(name)
            if match:
                angle_kind = match.group(1)
                return _apply_override(base_label, angle_kind)

            return base_label

        # convert param names to latex strings
        sanitized_names: list[str] = []
        latex_labels: list[str] = []
        corner_labels: list[str] = []
        seen_names: set[str] = set()

        for index, raw_name in enumerate(self.parameter_names):
            latex_label = _parameter_latex_label(raw_name)
            latex_label = _maybe_override_angle_label(raw_name, latex_label)
            sanitized_name = _sanitise_parameter_name(raw_name, index, seen_names)
            sanitized_names.append(sanitized_name)
            latex_labels.append(latex_label)
            corner_labels.append(f'${latex_label}$')

        if backend == 'corner':
            with matplotlib_latex():
                corner(
                    samples_for_corner,
                    **{
                        'labels': corner_labels,
                        'bins': 50,
                        'show_titles': True,
                        'title_fmt': '.3g',
                        'title_quantile': (0.025,0.5,0.975),
                        'quantiles': (0.025,0.5,0.975),
                        'smooth': 1,
                        'smooth1d': 1,
                        **kwargs
                    }
                )
                if save_path is not None:
                    plt.savefig(
                        save_path,
                        bbox_inches='tight',
                        dpi=300
                    )
        else:
            # Create a GetDist sample container for plotting with the paperplot style.
            samples_array = np.asarray(samples_for_corner, dtype=np.float64)

            # for longitude-like parameters, assume they have 'phi' in the name
            # and make sure getdist is aware of their periodicity
            periodic_ranges: dict[str, list[float | bool]] = {}
            for idx, sanitized_name in enumerate(sanitized_names):
                if 'phi' in sanitized_name.lower():
                    param_values = samples_array[:, idx]
                    max_val = float(np.max(param_values))

                    if normalised_coordinates is not None:
                        periodic_ranges[sanitized_name] = [0.0, 360.0, True]
                        assert max_val <= 360.
                    else:
                        periodic_ranges[sanitized_name] = [0.0, 2 * np.pi, True]
                        assert max_val <= 2 * np.pi

                    print(
                        f'Setting parameter at index {idx} ({sanitized_name}) '
                        f'to periodic {periodic_ranges[sanitized_name]}.'
                    )

            mc_samples = MCSamples(
                samples=samples_array,
                weights=self.weights,
                names=sanitized_names,
                labels=latex_labels,
                sampler='nested',
                ranges=periodic_ranges if periodic_ranges else None
            )
            mc_samples.updateSettings({'ignore_limits': True})

            plotter = plots.get_subplot_plotter(style=paperplot_style.style_name)

            default_triangle_options = {
                'filled': True,
                'legend_labels': None
            }
            default_triangle_options.update(kwargs)

            plotter.triangle_plot(
                [mc_samples],
                params=sanitized_names,
                **default_triangle_options
            )

            if save_path is not None:
                output_dir = os.path.dirname(save_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                plotter.export(save_path)

        plt.show()

    def corner_plot_double(self,
            second_model: Self,
            coordinates: list[str] | None = None,
            save_path: str | None = None,
            colors: list[str] = ['cornflowerblue', 'tomato'],
            labels: list[str] = ['Model 0', 'Model 1'],
            **corner_kwargs
        ) -> None:
        '''
        Make corner plot for NS run, with an additional NS overplotted.

        :param second_model: The second model to be over-plotted in the corner
            plot.
        :param coordinates: Specify a list of coordinates to transform the angle
            indices of the corner plot. If the list has two elements, the first
            coordinate is assumed to be the native coordinares and the last
            the target coordinates. For example, specifying
            `coordinates=['equatorial', 'galactic']` transforms from equatorial
            to galactic. Specifying `coordinates=['equatorial']` would leave
            the corner in its native coordinates, but since sampling is done
            internally in spherical coordinates, it would also involve a
            conversion to longitude and latitude in degrees.
        :param save_path: Specify a path to save the corner plot. If None, the
            plot will not be saved. The default is None.
        :param colors: Specify the colour of each model as an array, default is
            ['cornflowerblue', 'tomato'], where the first model is
            'cornflowerblue'.
        :param labels: Specify the corresponding model labels, default is
        ['Model 0', 'Model 1'], where the first model is 'Model 1'.
        '''
        fig = plt.figure(figsize=(10, 10))
        for model, color, label in zip([self, second_model], colors, labels):
            if coordinates is not None:
                samples_for_corner = model._convert_samples(model.samples, coordinates)
            else:
                samples_for_corner = model.samples
            
            corner(
                samples_for_corner,
                **{
                    'labels': model.parameter_names,
                    'bins': 50,
                    'show_titles': False,
                    'quantiles': (0.025,0.5,0.975),
                    'smooth': 1,
                    'smooth1d': 1,
                    'fig': fig,
                    'color': color,
                    **corner_kwargs
                }
            )
            plt.scatter([],[], color=color, label=label)
        plt.legend()
        if save_path is not None:
            plt.savefig(
                save_path,
                bbox_inches='tight',
                dpi=300
            )
        plt.show()

    def posterior_predictive_check(self,
            n_samples: int = 5,
            model_callable: Callable | None = None,
            **projview_kwargs
        ) -> None:
        '''
        Create a healpy posterior predictive map to heuristically verify the
        posterior distribution reflects the actual the data.

        :param n_samples: Number of posterior samples to draw.
        :param model_callable: If Posterior has been instantiated using a run
            number, the class will not know the model function transforming
            parameters to a healpy a map. Pass this function here.
        :param **projview_kwargs: Keyword arguments to pass to healpy's projview
            function.
        '''
        random_integers = np.random.randint(
            0,
            high=np.shape(self.samples)[0],
            size=n_samples
        )
        random_samples = self.samples[random_integers, :]
        
        try:
            predictive_maps = self.model(random_samples)
        
        except NotImplementedError:
            assert model_callable is not None, '''Please pass a callable model
function to this method when instantiating from an ultranest run number.'''
            predictive_maps = model_callable(random_samples)
        
        except Exception as e:
            raise Exception(e)
        
        # reconstruct map if pixels have been masked
        predictive_maps_for_projview = np.empty((self.npix, n_samples))
        predictive_maps_for_projview[self.boolean_mask, :] = predictive_maps
        predictive_maps_for_projview[~self.boolean_mask, :] = np.nan

        plt.figure(figsize=(4,9))  
        for i in range(n_samples):
            hp.projview(
                predictive_maps_for_projview[:, i],
                sub=(n_samples, 1, i+1), # type: ignore
                cbar=True,
                override_plot_properties={
                    'figure_width': 3
                },
                **projview_kwargs
            )
        plt.show()

    def sky_direction_posterior(self,
            coordinates: list[str] | None = None,
            save_path: str | None = None,
            colour: str = 'tomato',
            colours: Sequence[str] | None = None,
            smooth: None | float = 0.05,
            contour_levels: list[float] = [0.5, 1., 1.5, 2.],
            xsize: int = 500,
            nside: int = 256,
            rasterize_probability_mesh: bool = False,
            instantiate_new_axes: bool = True,
            label: str = 'Posterior Contours'
        ) -> None:
        '''
        :param coordinates: Specify a list of coordinates to transform the angle
            indices of the sky projection. If the list has two elements, the first
            coordinate is assumed to be the native coordinates and the last
            the target coordinates. For example, specifying
            `coordinates=['equatorial', 'galactic']` transforms from equatorial
            to galactic.
        :param save_path: Specify a path to save the corner plot. If None, the
            plot will not be saved. The default is None.
        :param colour: Legacy single-colour setting used when only one angular
            vector is present; also acts as the first fallback colour.
        :param colours: Optional sequence of colours to cycle through when plotting
            multiple angular parameter pairs.
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
        :param instantiate_new_axes: Choose whether or not to instantiate a
            blank set of Mollweide axes. If, for example, plotting multiple sky
            directions on the same axis, set this to True for the first
            call of sky_direction_posterior then False for subsequent calls.
        :param label: Label to display in the plot legend.
        '''
        angle_pairs = self._angle_parameter_pairs()
        if not angle_pairs:
            raise ValueError('No angular parameter pairs were found to plot.')

        base_samples = np.asarray(self.samples, dtype=np.float64)
        full_samples_for_sky = self._convert_samples(base_samples, coordinates)

        # build colour cycle ensuring at least as many distinct colours as pairs
        default_palette = [
            colour,
            'cornflowerblue',
            'mediumseagreen',
            'goldenrod',
            'mediumpurple',
            'darkcyan'
        ]
        if colours is not None:
            colour_cycle = list(colours)
            if not colour_cycle:
                colour_cycle = default_palette.copy()
        else:
            colour_cycle = default_palette.copy()

        # extend palette to cover all pairs without reusing the same colour immediately
        idx = 0
        while len(colour_cycle) < len(angle_pairs):
            colour_cycle.append(default_palette[idx % len(default_palette)])
            idx += 1

        if instantiate_new_axes:
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

        ax = plt.gca()
        existing_handles: list[Any] = []
        existing_labels: list[str] = []
        if not instantiate_new_axes:
            leg = ax.get_legend()
            if leg is not None:
                existing_handles = list(leg.legendHandles)
                existing_labels = [lab.get_text() for lab in leg.texts]

        handles = existing_handles
        labels_list = existing_labels

        def _format_angle_descriptor(param_name: str) -> str:
            '''Small helper to populate legend with which colour corresponds
            to which unit vector.'''
            match = _MULTIPOLE_ANGLE_PATTERN.match(param_name)
            if not match:
                return param_name
            suffix = match.group(2)
            ell_token, vec_token = suffix.split('_', maxsplit=1)
            ell_value = ell_token[1:]
            return rf'$\ell={ell_value}$ unit vector ({vec_token})'

        for pair_index, (phi_idx, theta_idx) in enumerate(angle_pairs):
            current_colour = colour_cycle[pair_index % len(colour_cycle)]
            longitude_deg = full_samples_for_sky[:, phi_idx]
            latitude_deg = full_samples_for_sky[:, theta_idx]

            probability_map = self._samples_to_healpy_map(
                longitude_deg,
                latitude_deg,
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

            phi_grid, theta_grid = np.meshgrid(phi, theta)
            transparent_cmap = self._make_transparent_colour_map(current_colour)

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
                colors=[matplotlib.colors.to_rgba(current_colour)], # type: ignore
                zorder=1,
                extend='both',
            )
            plt.pcolormesh(
                phi_grid,
                theta_grid,
                projected_map,
                cmap=transparent_cmap,
                rasterized=rasterize_probability_mesh,
            )

            if len(angle_pairs) == 1:
                descriptor = label if label else _format_angle_descriptor(
                    self.parameter_names[phi_idx]
                )
                pair_label = descriptor
            else:
                pair_label = _format_angle_descriptor(self.parameter_names[phi_idx])
            contour_proxy = Patch(
                facecolor=matplotlib.colors.to_rgba(current_colour, alpha=0.4),
                edgecolor=current_colour,
                linewidth=1,
                label=pair_label
            )
            handles.append(contour_proxy)
            labels_list.append(pair_label)

        ax.legend(handles=handles, labels=labels_list, loc='upper right')
        if save_path is not None:
            plt.savefig(
                save_path,
                bbox_inches='tight',
                dpi=300
            )

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

class Posterior(PosteriorMixin):
    def __init__(self,
            samples: NDArray[np.float64] | int | str,
            *,
            weights: NDArray[np.float64] | None = None
    ) -> None:
        '''
        :param samples: Can specify either an array of samples,
            an integer referring to a run number in ultranest_logs/ or a path
            to the ultranest log directory (the one containing the subdirs
            chains, extra, etc.). In the latter two cases, the samples from the
            run are automatically loaded. When providing arrays directly, pass
            either equal-weighted samples (leave ``weights`` as ``None``) or
            importance-weighted samples alongside their corresponding weights.
        '''
        self._parameter_names: list[str] = []
        self._weighted_samples: NDArray[np.float64] | None = None
        self._weights: NDArray[np.float64] | None = None

        if isinstance(samples, int):
            run_number = samples
            self._load_samples_from_log(run_number)
            self.loaded_from_run = True
        
        elif isinstance(samples, str):
            log_dir = samples
            self._load_samples_from_log(log_dir)
            self.loaded_from_run = True

        elif isinstance(samples, np.ndarray):
            sample_array = np.atleast_2d(np.asarray(samples, dtype=np.float64))
            if weights is not None:
                weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)
                if weights_array.shape[0] != sample_array.shape[0]:
                    raise ValueError(
                        'weights must have the same length as the number of samples.'
                    )
                if not np.any(weights_array > 0):
                    raise ValueError('weights must contain at least one positive entry.')
                resampled = resample_equal(sample_array, weights_array)
                self._samples = resampled
                self._weighted_samples = sample_array
                self._weights = weights_array
            else:
                self._samples = sample_array
                self._weighted_samples = sample_array
            self.loaded_from_run = False
        else:
            raise Exception('Pass either array of samples or an integer (run number).')

    @property
    def samples(self) -> NDArray[np.float64]:
        return self._samples
    
    @property
    def parameter_names(self) -> list[str]:
        return self._parameter_names

    def _load_samples_from_log(self,
            run_number: int | str
    ) -> None:
        if type(run_number) is int:
            base_dir = Path('ultranest_logs') / f'run{run_number}'
        else:
            assert type(run_number) is str
            base_dir = Path(run_number)

        # ultranest directory format
        chains_dir = base_dir / 'chains'
        info_dir = base_dir / 'info'
        equal_path = chains_dir / 'equal_weighted_post.txt'
        weighted_path = chains_dir / 'weighted_post.txt'
        info_path = str(info_dir / 'results.json')

        self._load_equal_weighted_samples(equal_path)
        self._load_weighted_samples(weighted_path)

        # extract marginal likelihood
        with open(info_path) as f:
            info = json.load(f)
            self.log_bayesian_evidence = info['logz']

    def _load_equal_weighted_samples(self, path: Path) -> None:
        self._samples = np.atleast_2d(
            np.loadtxt(path, skiprows=1, dtype=np.float64)
        )
        self._parameter_names = np.loadtxt(
            path,
            max_rows=1,
            dtype=str
        ).tolist()

    def _load_weighted_samples(self, path: Path) -> None:
        with path.open('r') as handle:
            header = handle.readline().strip().split()

        if len(header) < 3 or header[0].lower() != 'weight':
            raise ValueError(
                f'Weighted samples file {path} has unexpected header.'
            )

        raw_data = np.loadtxt(path, skiprows=1, dtype=np.float64)
        raw_data = np.atleast_2d(raw_data)

        if raw_data.shape[1] != len(header):
            raise ValueError(
                f'Mismatch between header and data columns in {path}.'
            )

        weights = np.asarray(raw_data[:, 0], dtype=np.float64)
        samples = np.atleast_2d(np.asarray(raw_data[:, 2:], dtype=np.float64))
        parameter_names = header[2:]

        if len(parameter_names) != samples.shape[1]:
            raise ValueError(
                'Number of parameter columns does not match parameter names '
                f'in {path}.'
            )

        if not hasattr(self, '_parameter_names') or not self._parameter_names:
            self._parameter_names = parameter_names
        elif list(self._parameter_names) != list(parameter_names):
            raise ValueError(
                'Parameter names in weighted chain do not match equal-weighted chain.'
            )

        self._weighted_samples = samples
        self._weights = weights

        # ensure equal-weighted samples exist by manually resampling with ultranest
        if not hasattr(self, '_samples') or getattr(self, '_samples') is None:
            self._samples = resample_equal(samples, weights)

class SKARun(Posterior):
    def __init__(self,
            briggs: Literal[-1, 0, 1],
            config: Literal['AA', 'AA4'],
            multiplier: Literal[1, 2],
            map_number: int,
            mask: Literal['full', 'northern', 'northern_galactic'],
            multipole_model: Literal[
                'monopole',
                'dipole',
                'dipole_quadrupole',
                'kinematic_dipole',
                'kinematic_dipole_quadrupole'
            ]
    ) -> None:
        if briggs == -1:
            self.briggs = 'n1'
        else:
            self.briggs = str(briggs)
        self.config = config
        self.multiplier = multiplier
        self.map_number = map_number
        self.mask = mask
        self.multipole_model = multipole_model

        self._make_file_path()
        self._load_samples_from_log(self.path)
    
    def _make_file_path(self):
        self.path = (
            f'output/ska/briggs_{self.briggs}/{self.config}'
            f'/mult_{self.multiplier}/map_{self.map_number}/{self.mask}'
            f'/{self.multipole_model}/run1'
        )
        assert os.path.exists(
            self.path
        ), f'Cannot find path ({self.path}).'
