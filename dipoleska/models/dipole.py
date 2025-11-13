from numpy.typing import NDArray
import numpy as np
from dipoleska.models.model_helpers import LikelihoodMixin, MapModelMixin
from dipoleska.utils.inference import InferenceMixin
from typing import Literal
from dipoleska.models.priors import Prior
from dipoleska.utils.math import compute_dipole_signal
from dipoleska.utils.posterior import PosteriorMixin

class Dipole(LikelihoodMixin, InferenceMixin, MapModelMixin, PosteriorMixin):
    def __init__(self,
            density_map: NDArray[np.int_ | np.float64],
            prior: Prior | None = None,
            likelihood: Literal['point', 'poisson', 'poisson_rms',
                                'general_poisson',
                                'general_poisson+rms'] = 'point',
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
            
            If no class is specified, the prior uses default dipole priors.
            These priors are specified in `models/default_priors.py`, with the
            following key exceptions:
            - if the Poisson likelihood is specified, the prior on the mean count
            parameter N will be automatically updated to a uniform distribution
            25% either side of the mean of the density map.
            - if the Poisson-RMS likelihood is specified, the prior will be
            updated to include both the mean count parameter N (with
            a uniform distribution 25% either side of the mean) and the
            RMS slope parameter (with a uniform distribution 25% either 
            side of the mean).
            If using a custom prior, the rms_slope parameter should appear at
            index 1 for now.

            In addition, if one specifies the point-by-point likelihood,
            the mean count parameter N is removed from the prior distribution
            and the dimensionality of the model is therefore reduced by 1.
        :param likelihood:
            Specify the type of likelihood function to use at inference:
            
            - `'poisson'` for the Poisson-based likelihood, or;
            - `'point'` for the point-by-point likelihood.
            - `'poisson_rms'` for the Poisson-based likelihood with 
                RMS augmentation.
            - `'general_poisson'` for the generalised Poisson likelihood.
            - `'general_poisson+rms'` for the generalised Poisson likelihood
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
        self._parse_prior_choice(default_prior='dipole', prior=prior)
        self._get_rms_fit_parameters(rms_map) 
        self._parse_likelihood_choice(likelihood)
        self._parameter_names = self.prior.parameter_names
        self.ndim = self.prior.ndim
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

        if self.likelihood == 'point':
            return self.point_by_point_log_likelihood(
                multipole_signal=dipole_term,
                density_map=self.density_map
            )
        if self.likelihood in ['poisson', 'poisson_rms']:
            return self.poisson_log_likelihood(
                rate_parameter=dipole_term,
                density_map=self.density_map
            )

        if self.likelihood in ['general_poisson', 'general_poisson_rms']:
            return self.general_poisson_log_likelihood(
                model_output=dipole_term,
                density_map=self.density_map
            )
        else:
            raise ValueError(
                f'Likelihood not recognised {self.likelihood}'
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
        
        if self.likelihood in ['poisson','general_poisson']:
            mean_count = Theta[:, 0]
            model_map = mean_count * (1 + dipole_signal)
            g_p = Theta[:, 1] if self.likelihood == 'general_poisson' else None
            return (model_map, g_p) if g_p is not None else model_map
        
        if self.likelihood in ['poisson_rms', 'general_poisson_rms']:
            mean_count = Theta[:, 0]
            rms_slope = Theta[:, 1]
            
            # (n_pix, )
            assert self.rms_map is not None
            rms_ratio = self.rms_map/self.rms_ref
            
            # (n_pix, 1) * (1, n_live) --> (n_pix, n_live)
            rms_scaling = rms_ratio[:, None] ** (-rms_slope[None, :])

            # (1, n_live) * (n_pix, n_live) * (n_pix, n_live )--> (n_pix, n_live)
            model_map = mean_count[None, :] * rms_scaling * (1 + dipole_signal)
            
            g_p = Theta[:, 2] if self.likelihood == 'general_poisson_rms' else None
            return (model_map, g_p) if g_p is not None else model_map
            
            
        
        # if self.likelihood == 'poisson':
        #     mean_count = Theta[:, 0]
        #     model_map = mean_count * (1 + dipole_signal)
        #     return model_map
        
        # if self.likelihood == 'general_poisson':
        #     mean_count = Theta[:, 0]
        #     glb_param = Theta[:, 1]
        #     model_map = mean_count * (1 + dipole_signal)
        #     return (model_map, glb_param)

        # if self.likelihood == 'poisson_rms':
        #     mean_count = Theta[:, 0]
        #     rms_slope = Theta[:, 1]
            
        #     # (n_pix, )
        #     assert self.rms_map is not None
        #     rms_ratio = self.rms_map/self.rms_ref
            
        #     # (n_pix, 1) * (1, n_live) --> (n_pix, n_live)
        #     rms_scaling = rms_ratio[:, None] ** (-rms_slope[None, :])

        #     # (1, n_live) * (n_pix, n_live) * (n_pix, n_live )--> (n_pix, n_live)
        #     model_map = mean_count[None, :] * rms_scaling * (1 + dipole_signal)
        #     return model_map
        
        # if self.likelihood == 'general_poisson_rms':
        #     mean_count = Theta[:, 0]
        #     rms_slope = Theta[:, 1]
        #     glb_param = Theta[:, 2]
            
        #     # (n_pix, )
        #     assert self.rms_map is not None
        #     rms_ratio = self.rms_map/self.rms_ref
            
        #     # (n_pix, 1) * (1, n_live) --> (n_pix, n_live)
        #     rms_scaling = rms_ratio[:, None] ** (-rms_slope[None, :])

        #     # (1, n_live) * (n_pix, n_live) * (n_pix, n_live )--> (n_pix, n_live)
        #     model_map = mean_count[None, :] * rms_scaling * (1 + dipole_signal)
        #     return (model_map, glb_param)
        
        else:
            raise ValueError(
                f'Likelihood not recognised {self.likelihood}'
            )

