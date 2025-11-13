from numpy.typing import NDArray
import numpy as np
from dipoleska.models.model_helpers import LikelihoodMixin, MapModelMixin
from dipoleska.utils.inference import InferenceMixin
from typing import Literal, cast
from dipoleska.models.priors import Prior
from dipoleska.utils.math import compute_dipole_signal, vectorised_rms_signal
from dipoleska.utils.posterior import PosteriorMixin

class Dipole(LikelihoodMixin, InferenceMixin, MapModelMixin, PosteriorMixin):
    def __init__(self,
            density_map: NDArray[np.int_ | np.float64],
            prior: Prior | None = None,
            likelihood: Literal['point', 'poisson', 'poisson_rms',
                                'general_poisson',
                                'general_poisson_rms'] = 'point',
            rms_map: NDArray[np.float64] | None = None,
            fixed_dipole: tuple[float, float, float] | None = None
    ):
        '''
        A pure dipole model. Depending on the choice of likelihood function,
        this model has 3 or 4 parameters (see below). Note that this model is
        vectorized by default in the sense that it can handle calls from the
        Nested Sampler with many live points at once.

        :param density_map:
            Healpy source density map, of shape (n_pix,).
            Masked pixels (if any) should be filled with np.nan,
            which are then automatically masked for the likelihood evaluation.
        :param prior:
            Pass either an instance of a Prior object or leave as None.

            - If ``None`` is provided, the model builds a likelihood-specific
              default prior using the canonical parameter names below:
                * ``Nbar`` — mean count (Poisson/General Poisson likelihoods)
                * ``rms_slope`` — RMS scaling slope (``*_rms`` likelihoods)
                * ``gp_dispersion`` — generalised Poisson dispersion
                * ``D`` — dipole amplitude
                * ``phi`` — dipole longitude
                * ``theta`` — dipole colatitude
            - If a Prior instance is supplied, any matching names override the
              defaults while unsupplied parameters retain these built-in choices.
              Parameter names must come from the list above; unrecognised names
              raise a ValueError. No ordering constraints apply—parameters are
              accessed by name internally.
        :param likelihood:
            Specify the type of likelihood function to use at inference:
            
            - `'poisson'` for the Poisson-based likelihood, or;
            - `'point'` for the point-by-point likelihood.
            - `'poisson_rms'` for the Poisson-based likelihood with 
                RMS augmentation.
            - `'general_poisson'` for the generalised Poisson likelihood.
            - `'general_poisson_rms'` for the generalised Poisson likelihood
                with RMS augmentation.
            As mentioned above, this choice will dynamically change the model
            dimensionality and the prior distributions.
        :param rms_map:
            RMS pointings map of shape (n_pix,). Required if the `poisson_rms` 
            likelihood is specified. Masked pixels (if any) should be filled 
            with np.nan, which are then automatically masked for the likelihood 
            evaluation.
        :param fixed_dipole: Specify a tuple containing a dipole amplitude,
            longitude in radians and colatitude in radians in order. The model
            then fits the sum of two dipoles: this specified (fixed) dipole
            plus a free (fitted) dipole.
        '''
        self._get_healpy_map_attributes(density_map)
        self._get_rms_fit_parameters(rms_map)
        self._setup_dipole_prior(prior=prior, likelihood=likelihood)
        self._parameter_names = self.prior.parameter_names
        self.ndim = self.prior.ndim
        self._cache_parameter_indices()
        if fixed_dipole is not None:
            self.fixed_dipole = np.asarray(fixed_dipole)
        else:
            self.fixed_dipole = None

    @property
    def density_map(self) -> NDArray[np.int_ | np.float64]:
        '''
        Whenever the model calls the `density_map` attribute, provide only the
        unmasked pixels for inference.
        '''
        return self._density_map[self.boolean_mask]
    
    @property
    def rms_map(self) -> NDArray[np.int_ | np.float64] | None:
        '''
        Whenever the model calls the `rms_map` attribute, provide only the
        unmasked pixels for inference.
        '''
        if self._rms_map is None:
            return
        else:
            return self._rms_map[self.boolean_mask]
    
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

        # we use casts to make explicit (and for type checkers) exactly
        # what shape arrives at each branch
        if self.likelihood == 'point':
            dipole_signal = cast(NDArray[np.float64], dipole_term)
            return self.point_by_point_log_likelihood(
                multipole_signal=dipole_signal,
                density_map=self.density_map
            )

        if self.likelihood in ['poisson', 'poisson_rms']:
            rate_parameter = cast(NDArray[np.float64], dipole_term)
            return self.poisson_log_likelihood(
                rate_parameter=rate_parameter,
                density_map=self.density_map
            )

        if self.likelihood in ['general_poisson', 'general_poisson_rms']:
            model_map, gp_dispersion = cast(
                tuple[NDArray[np.float64], NDArray[np.float64]],
                dipole_term
            )
            return self.general_poisson_log_likelihood(
                model_output=(model_map, gp_dispersion),
                density_map=self.density_map
            )

        else:
            raise ValueError(
                f'Likelihood not recognised {self.likelihood}'
            )

    def _cache_parameter_indices(self) -> None:
        '''
        Cache parameter indices for fast lookup from Theta by name.
        '''
        self._parameter_indices = {
            name: self.prior.index_for(name)
            for name in self.prior.parameter_names
        }

    def _theta_param(self, Theta: NDArray[np.float64], name: str) -> NDArray[np.float64]:
        idx = self._parameter_indices[name]
        return Theta[:, idx]

    def _optional_theta_param(
            self,
            Theta: NDArray[np.float64],
            name: str
    ) -> NDArray[np.float64] | None:
        idx = self._parameter_indices.get(name)
        if idx is None:
            return None
        return Theta[:, idx]

    def _setup_dipole_prior(
            self,
            prior: Prior | None,
            likelihood: Literal['point', 'poisson', 'poisson_rms',
                                'general_poisson', 'general_poisson_rms']
    ) -> None:
        '''
        Build the prior dictionary for the requested likelihood, allowing
        user-specified priors to override individual parameters while falling
        back to defaults for everything else.
        '''
        if likelihood in ['poisson_rms', 'general_poisson_rms']:
            assert self._rms_map is not None, (
                f"rms_map must be provided when using '{likelihood}' likelihood."
            )

        self.likelihood = likelihood
        default_prior_dict = self._default_prior_aliases(likelihood=likelihood)

        if prior is None:
            self._prior = Prior(choose_prior=default_prior_dict)
            return

        assert hasattr(prior, 'prior_dict'), (
            'Custom priors must expose a prior_dict attribute.'
        )
        user_dict = prior.prior_dict
        unknown = sorted(set(user_dict) - set(default_prior_dict))
        if unknown:
            raise ValueError(
                'Unrecognised prior parameter(s) for Dipole model: '
                + ', '.join(unknown)
            )

        merged = default_prior_dict.copy()
        merged.update(user_dict)
        defaulted = sorted(set(default_prior_dict) - set(user_dict))
        if defaulted:
            print(
                '[Dipole] Using default priors for parameters: '
                + ', '.join(defaulted)
            )

        self._prior = Prior(choose_prior=merged)

    def _default_prior_aliases(
            self,
            likelihood: Literal['point', 'poisson', 'poisson_rms',
                                'general_poisson', 'general_poisson_rms']
    ) -> dict[str, list[float | np.floating | str]]:
        '''
        Assemble the default prior dictionary for the requested likelihood.
        '''
        defaults: dict[str, list[float | np.floating | str]] = {}
        if likelihood != 'point':
            defaults['Nbar'] = [
                'Uniform',
                0.75 * self.mean_density,
                1.25 * self.mean_density
            ]

        if 'rms' in likelihood:
            defaults['rms_slope'] = [
                'Uniform',
                0.75 * self.rms_slope,
                1.25 * self.rms_slope
            ]

        if 'general_poisson' in likelihood:
            defaults['gp_dispersion'] = ['Uniform', 0.0, 1.0]

        defaults.update({
            'D': ['Uniform', 0.0, 0.1],
            'phi': ['Uniform', 0.0, 2 * np.pi],
            'theta': ['Polar', 0.0, np.pi]
        })
        return defaults

    def model(self,
            Theta: NDArray[np.float64]
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''
        Evaluates 1 + D cos(theta) for the dipole model.
        
        :param Theta: Model prior samples, of shape (n_live, n_dim).
        :return: Vectorised evaluation of 1 + D cos(theta), of shape
            (n_pix, n_live).
        '''
        dipole_amplitude = self._theta_param(Theta, 'D')
        dipole_longitude = self._theta_param(Theta, 'phi')
        dipole_colatitude = self._theta_param(Theta, 'theta')

        dipole_signal = compute_dipole_signal(
            dipole_amplitude=dipole_amplitude,
            dipole_longitude=dipole_longitude,
            dipole_colatitude=dipole_colatitude,
            pixel_vectors=self.pixel_vectors
        )

        if self.fixed_dipole is not None:
            # reshape fixed dipole such that it is as long as n_live
            n_live = Theta.shape[0]
            fixed_amplitude = np.full(n_live, self.fixed_dipole[0])
            fixed_longitude = np.full(n_live, self.fixed_dipole[1])
            fixed_colatitude = np.full(n_live, self.fixed_dipole[2])
            
            dipole_signal += compute_dipole_signal(
                fixed_amplitude,
                fixed_longitude,
                fixed_colatitude,
                pixel_vectors=self.pixel_vectors
            )

        if self.likelihood == 'point':
            return 1 + dipole_signal
        
        if self.likelihood in ['poisson', 'general_poisson']:
            mean_count = self._theta_param(Theta, 'Nbar')
            model_map = mean_count * (1 + dipole_signal)

            if self.likelihood == 'general_poisson':
                gp_disperson = self._theta_param(Theta, 'gp_dispersion')
                return (model_map, gp_disperson)
            else:
                return model_map
        
        if self.likelihood in ['poisson_rms', 'general_poisson_rms']:
            mean_count = self._theta_param(Theta, 'Nbar')
            rms_slope = self._theta_param(Theta, 'rms_slope')
            
            assert self.rms_map is not None
            rms_ratio = self.rms_map / self.rms_ref
            rms_scaling = vectorised_rms_signal(rms_ratio, rms_slope)
            
            # (1, n_live) * (n_pix, n_live) * (n_pix, n_live )--> (n_pix, n_live)
            model_map = mean_count[None, :] * rms_scaling * (1 + dipole_signal)

            if self.likelihood == 'general_poisson_rms':
                gp_disperson = self._theta_param(Theta, 'gp_dispersion')
                return (model_map, gp_disperson)
            else:
                return model_map
        
        else:
            raise ValueError(
                f'Likelihood not recognised {self.likelihood}'
            )
