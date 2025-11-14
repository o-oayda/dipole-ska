import healpy as hp
import numpy as np
from numpy.typing import NDArray
from typing import Literal
from scipy.stats import poisson
from scipy.special import gammaln
from abc import abstractmethod
from dipoleska.models.priors import Prior
from dipoleska.utils.math import rms_power_law_fit

class LikelihoodMixin:
    @property
    @abstractmethod
    def prior(self) -> Prior:
        raise NotImplementedError('Subclass models must implement a prior property.')

    def point_by_point_log_likelihood(self,
            multipole_signal: NDArray[np.float64],
            density_map: NDArray[np.int_ | np.float64]
    ) -> NDArray[np.float64]:
        '''
        Compute the vectorised log likelihood for many dipole maps using
        the point-by-point function.

        :param multipole_signal: Array of dipole signals, defined as
            f_dipole = 1 + D cos ( theta ), or potentially multipole signals.
            The shape of the array should be (n_pixels, n_live), where
            n_live is the number of live points used in ultranest's
            vetcorised function call and n_pixels is the number of pixels
            in the healpy map.
        :param density_map: Healpy density map of shape (n_pixels,).
        :return: Log likelihood corresponding to each dipole signal of shape
            (n_live,).
        '''
        normalisation_factor = np.sum(multipole_signal, axis=0)
        likelihood_map = multipole_signal / normalisation_factor
        log_likelihood = np.einsum(
            'i,ij->j',
            density_map,
            np.log(likelihood_map)
        )
        return log_likelihood

    def poisson_log_likelihood(self,
            rate_parameter: NDArray[np.float64],
            density_map: NDArray[np.int_ | np.float64]
    ) -> NDArray[np.float64]:
        '''
        Compute the vectorised log likelihood for many dipole maps using the
        Poisson likelihood function.

        :param rate_parameter: Array of rate parameters for each cell; of
            shape (n_pixels, n_live).
        :param density_map: Healpy density map of shape (n_pixels,).
        :return: Log likelihood corresponding to each dipole signal of shape
            (n_live,).
        '''
        log_likelihood_map = poisson.logpmf(
            k=density_map[:, None],
            mu=rate_parameter
        )
        return np.sum(log_likelihood_map, axis=0)
    
    def general_poisson_log_likelihood(self,
            model_output: tuple[NDArray[np.float64], NDArray[np.float64]],
            density_map: NDArray[np.int_ | np.float64]
    ) -> NDArray[np.float64]:
        '''
        Compute the vectorised log likelihood for many dipole maps using the
        generalised Poisson likelihood function.

        :param model_output: Tuple containing the outputs from the generalised
            Poisson model. The first element is an array of rate parameters lambda,
            standard for the Poisson distribuion, of shape (n_pixels, n_live).
            The second is an array of generalised Poisson dispersion parameters b_GP
            (see (24) in vonHausegger+25), of shape (n_live,). Thus the tuple is:
            (lambda, b_GP).
        :param density_map: Healpy density map of shape (n_pixels,).
        :return: Log likelihood corresponding to each dipole signal of shape
            (n_live,).
        '''
        rate_parameter, b = model_output
        b = b[None, :]  #(1, n_live)

        term1 = np.log(rate_parameter * (1 - b))                        
        term2 = (density_map[:, None] - 1) * np.log(rate_parameter * (1 - b) + 
                                                    density_map[:, None] * b)
        term3 = gammaln(density_map + 1)[:, None]                      
        term4 = rate_parameter * (1 - b)
        term5 = density_map[:, None] * b

        logL_pixels = term1 + term2 - term3 - term4 - term5  # (n_pixels, n_live)
        
        logL = np.sum(logL_pixels, axis=0)  # shape (n_live,)
        
        return logL


    def prior_transform(self,
            uniform_deviates: NDArray[np.float64]
        ) -> NDArray[np.float64]:
        '''
        Meta function passed to the Nested Sampler's prior input; calls
        `Prior` object's `transform` method to turn deviates on the unit cube
        to deviates in prior space.
        '''
        return self.prior.transform(uniform_deviates)


class MapModelMixin:
    @property
    @abstractmethod
    def density_map(self) -> NDArray[np.float64 | np.int_]:
        raise NotImplementedError('Subclass models must define a density map.')

    def _get_healpy_map_attributes(self,
            density_map: NDArray[np.int_ | np.float64]
    ) -> None:
        '''
        Retrieve key properties of a healpix map.
        '''
        self.mean_density = np.nanmean(density_map)
        self.nside = hp.get_nside(density_map)
        self.npix = hp.nside2npix(self.nside)
        pixels_x, pixels_y, pixels_z = hp.pix2vec(
            self.nside,
            np.arange(self.npix)
        )
        self._pixel_vectors = np.stack([pixels_x, pixels_y, pixels_z]).T # (n_pix, 3)
        self._density_map = density_map

        # masked attributes
        self.boolean_mask = ~np.isnan(density_map)
        self.n_unmasked = np.sum(self.boolean_mask, dtype=np.int64)
        self._density_map_masked = self._density_map[self.boolean_mask]
        self._pixel_vectors_masked = self._pixel_vectors[self.boolean_mask]
        x, y, z = self._pixel_vectors[self.boolean_mask].T
        self._pixel_vectors_xyz_masked = [x, y, z]

    @staticmethod
    def _log_prior_sources(
            model_label: str,
            merged: dict[str, list[float | np.floating | str]],
            overrides: dict[str, list[float | np.floating | str]]
    ) -> None:
        lines = MapModelMixin._prior_configuration_lines(
            model_label=model_label,
            merged=merged,
            overrides=overrides
        )
        print('\n'.join(lines))

    @staticmethod
    def _prior_configuration_lines(
            model_label: str,
            merged: dict[str, list[float | np.floating | str]],
            overrides: dict[str, list[float | np.floating | str]]
    ) -> list[str]:
        lines = [f'[{model_label}] Prior configuration:']
        for name, alias in merged.items():
            source = 'custom' if name in overrides else 'default'
            lines.append(
                f'  - {name}: {MapModelMixin._format_alias(alias)} ({source})'
            )
        return lines

    @staticmethod
    def _format_alias(alias: list[float | np.floating | str]) -> str:
        formatted = []
        for item in alias:
            if isinstance(item, (float, np.floating)):
                formatted.append(f'{item:.4g}')
            else:
                formatted.append(str(item))
        return '[' + ', '.join(formatted) + ']'
        
    def _get_rms_fit_parameters(self,
            rms_map: NDArray[np.float64] | None
            ) -> None:
        '''
        Fit the rms map to a power law and store the fit parameters.

        :param rms_map: 1D numpy array of shape (n_pix,). Must have the same 
            shape, pixel ordering, and HEALPix nside as the density map used 
            in this model. Both maps should use the same masking convention 
            (np.nan for masked pixels), and be aligned such that each element 
            corresponds to the same sky pixel.
        '''
        self._rms_map = rms_map
        if self._rms_map is not None:
            self.rms_ref = np.nanmedian(self._rms_map)
            self._rms_map_masked = self._rms_map[self.boolean_mask]
            self.rms_mean_density, self.rms_slope = rms_power_law_fit(
                rms_map=self._rms_map_masked,
                density_map=self._density_map_masked
            )

    def _parse_likelihood_choice(self,
            likelihood: Literal['point', 'poisson', 'poisson_rms',
                                'general_poisson', 'general_poisson_rms']
        ) -> None:
        '''
        If one specifies the point-by-point likelihood, we don't need to fit
        for the mean density. This function removes that parameter from the
        list of priors and parameter names, reducing the dimension of the model
        by 1.

        In addition, if one chooses the Poisson likelihood, we ideally
        want the choice of mean density prior to center around the mean density
        itself. This automatically makes that change without needing explicit
        input from the user.
        
        If one chooses the 'poisson_rms' likelihood, we fit for both
        the mean density and slope of the rms-power-law relation, and update
        the priors accordingly.
        
        Finally, if one chooses the 'general_poisson' likelihood, we add
        an additional prior for the generalised Poisson parameter.
        
        Prior ordering will be: 
        (mean_count, rms_slope (if applicable), gp_dispersion (if applicable), ....)
        '''
        self.likelihood = likelihood
        
        if self.likelihood == 'point':
            self._prior.remove_prior(prior_indices=[0])

        elif self.likelihood in ['poisson', 'poisson_rms',
                                 'general_poisson', 'general_poisson_rms']:
            self._prior.change_prior(
                prior_index=0,
                new_prior=[
                    'Uniform',
                    0.75 * self.mean_density,
                    1.25 * self.mean_density
                ]
            )
            if self.likelihood in ['poisson_rms','general_poisson_rms']:
                assert self._rms_map is not None, (
                    f"rms_map must be provided when using "
                    f"'{self.likelihood}' likelihood."
                )
                # if (
                # self.likelihood in ['poisson_rms','general_poisson_rms']
                # and 'rms_slope' not in getattr(model.prior, "prior_dict", {})
                # ):
                self._prior.add_prior(
                    prior_index=1,
                    prior_name='rms_slope',
                    prior_alias=[
                        'Uniform',
                        0.75 * self.rms_slope,
                        1.25 * self.rms_slope
                    ]
                )
                
            if self.likelihood == 'general_poisson':
                # if (
                # self.likelihood == 'general_poisson'
                # and 'glb_param' not in getattr(model.prior, "prior_dict", {})
                # ):
                    self._prior.add_prior(
                        prior_index=1,
                        prior_name='gp_dispersion',
                        prior_alias=['Uniform',0.,1.]
                        )
                
            if self.likelihood == 'general_poisson_rms':
                # if (
                # self.likelihood == 'general_poisson_rms'
                # and 'glb_param' not in getattr(model.prior, "prior_dict", {})
                # ):
                self._prior.add_prior(
                    prior_index=2,
                    prior_name='gp_dispersion',
                    prior_alias=['Uniform',0.,1.]
                    )

        else:
            raise Exception(
                f'Likelihood choice ({self.likelihood}) not recognised.'
            )
