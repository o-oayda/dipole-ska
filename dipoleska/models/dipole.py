from numpy.typing import NDArray
import numpy as np
from scipy.stats import poisson
from dipoleska.utils.inference import InferenceMixin
from typing import Literal
from dipoleska.models.priors import Prior
import healpy as hp
from dipoleska.utils.math import compute_dipole_signal
from dipoleska.utils.posterior import PosteriorMixin
from abc import abstractmethod

class LikelihoodMixin:
    @property
    @abstractmethod
    def prior(self) -> Prior:
        raise NotImplementedError('Subclass models must implement a prior property.')

    def point_by_point_log_likelihood(self,
            dipole_signal: NDArray[np.float64],
            density_map: NDArray[np.int_ | np.float64]
    ) -> NDArray[np.float64]:
        '''
        Compute the vectorised log likelihood for many dipole maps using
        the point-by-point function.

        :param dipole_signal: Array of dipole signals, defined as
            f_dipole = 1 + D cos ( theta ).
            The shape of the array should be (n_pixels, n_live), where
            n_live is the number of live points used in ultranest's
            vetcorised function call and n_pixels is the number of pixels
            in the healpy map.
        :param density_map: Healpy density map of shape (n_pixels,).
        :return: Log likelihood corresponding to each dipole signal of shape
            (n_live,).
        '''
        normalisation_factor = np.sum(dipole_signal, axis=0)
        likelihood_map = dipole_signal / normalisation_factor
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
        self.boolean_mask = ~np.isnan(density_map)
    
    def _parse_prior_choice(self,
            default_prior: str,
            prior: Prior | None = None
        ) -> None:
        '''
        Switch to a default prior if the user has not specified one, or use
        the explicit one the user has provided.
        '''
        if prior is None:
            self._prior = Prior(choose_prior=default_prior)
        else:
            self._prior = prior
    
    def _parse_likelihood_choice(self,
            likelihood: Literal['point', 'poisson']
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
        '''
        self.likelihood = likelihood

        if self.likelihood == 'point':
            self._prior.remove_prior(prior_index=0)
        elif self.likelihood == 'poisson':
            self._prior.change_prior(
                prior_index=0,
                new_prior=[
                    'Uniform',
                    0.75 * self.mean_density,
                    1.25 * self.mean_density
                ]
            )
        else:
            raise Exception(
                f'Likelihood choice ({self.likelihood}) not recognised.'
            )

class Dipole(LikelihoodMixin, InferenceMixin, MapModelMixin, PosteriorMixin):
    def __init__(self,
            density_map: NDArray[np.int_ | np.float64],
            prior: Prior | None = None,
            likelihood: Literal['point', 'poisson'] = 'point',
    ):
        '''
        A pure dipole model. Depending on the choice of likelihood function,
        this model has 3 or 4 parameters (see below). Note that this model is
        vectorized by default in the sense that it can handle calls from the
        Nested Sampler with many live points at once.

        :param density_map:
            Healpy source density map, of shape (n_pix,).
        :param prior:
            Pass either an instance of a Prior object or leave as None.
            
            If no class is specified, the prior uses default dipole priors.
            These priors are specified in `models/default_priors.py`, with one
            key exception: if the Poisson likelihood is specified, the
            prior on the mean count parameter N will be automatically updated
            to a uniform distribuion 25% either side of the mean of the density
            map.
            
            In addition, if one specifies the point-by-point likelihood,
            the mean count parameter N is removed from the prior distribution
            and the dimensionality of the model is therefore reduced by 1.
        :param likelihood:
            Specify the type of likelihood function to use at inference:
            
            - `'poisson'` for the Poisson-based likelihood, or;
            - `'point'` for the point-by-point likelihood.
            
            As mentioned above, this choice will dynamically change the model
            dimensionality and the prior distributions.
        '''
        self._get_healpy_map_attributes(density_map)
        self._parse_prior_choice(default_prior='dipole', prior=prior)
        self._parse_likelihood_choice(likelihood)
        self._parameter_names = self.prior.parameter_names
        self.ndim = self.prior.ndim

    @property
    def density_map(self) -> NDArray[np.int_ | np.float64]:
        '''
        Whenever the model calls the `density_map` attribute, provide only the
        unmasked pixels for inference.
        '''
        return self._density_map[self.boolean_mask]
    
    @property
    def pixel_vectors(self) -> NDArray[np.float64]:
        '''
        Whenever the model calls the `pixel_vectors` attribute, provide only
        the vectors pointing to unmasked pixels.
        '''
        return self._pixel_vectors[self.boolean_mask]

    @property
    def prior(self) -> Prior:
        return self._prior
    
    @property
    def parameter_names(self) -> list[str]:
        return self._parameter_names

    def log_likelihood(self,
            Theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        '''
        Log likelihood function passed to the Nested Sampler. Automatically
        adjusted depending on the user-specified prior.

        :param Theta: Parameter samples from the prior distribtuion, as
            generated by the Nested Sampler.
        '''
        dipole_term = self.model(Theta)

        if self.likelihood == 'point':
            return self.point_by_point_log_likelihood(
                dipole_signal=dipole_term,
                density_map=self.density_map
            )
        else:
            return self.poisson_log_likelihood(
                rate_parameter=dipole_term,
                density_map=self.density_map
            )

    def model(self,
            Theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        '''
        Evaluates 1 + D cos(theta) for the dipole model.
        
        :param Theta: Model prior samples, of shape (n_live, n_dim).
        :return: Vectorised evaluation of 1 + D cos(theta), of shape
            (n_pix, n_live).
        '''
        dipole_amplitude = Theta[:, -3]
        dipole_longitude = Theta[:, -2]
        dipole_colatitude = Theta[:, -1]

        dipole_signal = compute_dipole_signal(
            dipole_amplitude=dipole_amplitude,
            dipole_longitude=dipole_longitude,
            dipole_colatitude=dipole_colatitude,
            pixel_vectors=self.pixel_vectors
        )

        if self.likelihood == 'point':
            return 1 + dipole_signal
        else:
            mean_count = Theta[:, 0]
            return mean_count * (1 + dipole_signal)