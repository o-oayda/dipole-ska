from numpy.typing import NDArray
import numpy as np
from collections import defaultdict
from dipoleska.utils.inference import InferenceMixin
from dipoleska.utils.math import (
    compute_dipole_signal, multipole_pixel_product_vectorised,
    multipole_tensor_vectorised, vectorised_quadrupole_tensor, vectorised_rms_signal,
    vectorised_spherical_to_cartesian
)
from dipoleska.utils.posterior import PosteriorMixin
from dipoleska.models.model_helpers import LikelihoodMixin, MapModelMixin
from dipoleska.models.priors import Prior

class Multipole(LikelihoodMixin, InferenceMixin, MapModelMixin, PosteriorMixin):
    def __init__(self,
            density_map: NDArray[np.int_ | np.float64],
            ells: list[int],
            prior: Prior | None = None, 
            rms_map: NDArray[np.float64] | None = None
    ):
        '''
        Fit an abitrary number of monopoles of different orders.

        :param density_map:
            Healpy source density map, of shape (n_pix,).
        :param prior:
            Pass either an instance of a Prior object or leave as None.

            - If ``None`` is specified, the model constructs a default prior
              covering every amplitude ``Mℓ`` and the ``2ℓ`` angular parameters
              for each multipole order in ``ells``. The recognised names are:
                * ``Mℓ`` for each amplitude (e.g. ``M0``, ``M1``, ``M2``)
                * ``phi_lℓ_k`` / ``theta_lℓ_k`` for the ``k``-th unit vector of
                  order ``ℓ`` (e.g. ``phi_l2_1``, ``theta_l3_0``)
            - If a Prior instance is supplied, any matching names override the
              defaults while unsupplied parameters keep their built-in
              distributions. Names must follow the convention above;
              unrecognised names raise a ValueError. Ordering is irrelevant, as
              parameters are accessed by name internally.
        :param ells:
            Pass a list of multipole orders to fit, e.g. `ells = [1, 2, 3]`.
            If a monopole (0) is specified in the list, the Poissonian
            likelihood is used; else, the point-by-point likelihood is used.
        
        TODO: performance is still slower than dipole-stats implementation;
            determine why this is.
            dipole-stats: 116s
            dipole-ska: 158s
            - note 11.11.25: performance looks to be only a ~10% difference
                judging from the fiducial map integration test.
        '''
        self._get_healpy_map_attributes(density_map)
        self._get_rms_fit_parameters(rms_map)
        self.ells = ells
        self.multipole_orders = [ell for ell in ells if ell != 0]
        self._setup_multipole_prior(ells=ells, prior=prior)
        
        # if we are fitting a monopole (ell=0), adjust the monopole prior to center
        # around the mean number density; otherwise, for point-by-point, the
        # prior will automatically lack a monopole prior, so no change needed
        if any(ell == 0 for ell in ells):
            self.monopole_is_fitted = True
            if self._rms_map is None:
                likelihood = 'poisson'
            else:
                likelihood = 'poisson_rms'
            self._parse_likelihood_choice(likelihood)
        else:
            self.monopole_is_fitted = False
            assert self._rms_map is None, (
                "When passing an rms map (Poisson rms likelihood), "
                "0 must be included in the list of ells."
            )

        self._parameter_names = self.prior.parameter_names
        self.ndim = self.prior.ndim
        self.n_multipoles = len(self.multipole_orders)
        self._cache_parameter_indices()
        self._get_angle_indices()
    
    def _setup_multipole_prior(
            self,
            ells: list[int],
            prior: Prior | None
    ) -> None:
        '''
        Build priors for arbitrary ells, allowing user-specified entries to
        override default choices.
        '''
        default_prior_dict = self._default_multipole_prior_aliases(ells)

        if prior is None:
            self._log_prior_sources('Multipole', default_prior_dict, overrides={})
            self._prior = Prior(choose_prior=default_prior_dict)
            return

        assert hasattr(prior, 'prior_dict'), (
            'Custom priors must expose a prior_dict attribute.'
        )
        user_dict = prior.prior_dict
        unknown = sorted(set(user_dict) - set(default_prior_dict))
        if unknown:
            raise ValueError(
                'Unrecognised prior parameter(s) for Multipole model: '
                + ', '.join(unknown)
            )

        merged = default_prior_dict.copy()
        merged.update(user_dict)
        self._log_prior_sources('Multipole', merged, user_dict)

        self._prior = Prior(choose_prior=merged)

    def _default_multipole_prior_aliases(
            self,
            ells: list[int]
    ) -> dict[str, list[float | np.floating | str]]:
        azimuthal = ['Uniform', 0.0, 2 * np.pi]
        polar = ['Polar', 0.0, np.pi]
        priors: dict[str, list[float | np.floating | str]] = {}

        for ell in ells:
            amplitude_name = f'M{ell}'
            if ell == 0:
                priors[amplitude_name] = [
                    'Uniform',
                    0.75 * self.mean_density,
                    1.25 * self.mean_density
                ]
            else:
                priors[amplitude_name] = ['Uniform', 0.0, 0.1 * ell ** 2]

        for ell in ells:
            if ell == 0:
                continue
            for vec_idx in range(ell):
                priors[f'phi_l{ell}_{vec_idx}'] = azimuthal.copy()
                priors[f'theta_l{ell}_{vec_idx}'] = polar.copy()

        return priors


    def _get_angle_indices(self):
        self.phi_indices = defaultdict(list)
        self.theta_indices = defaultdict(list)
        
        for i, key in enumerate(self.parameter_names):
            if 'rms' in key: # skip over rms slope param
                continue
            elif 'M' in key:
                continue
            else:
                angle_type, ell, vec_number = key.split('_') # e.g. phi_l2_1

                new_key = f'{ell[1:]}'

                if angle_type == 'phi':
                    self.phi_indices[new_key].append(i)
                else:
                    self.theta_indices[new_key].append(i)

    def _parse_multipole_likelihood(self, ells: list[int]) -> None:
        pass

    @property
    def density_map(self) -> NDArray[np.int_ | np.float64]:
        '''
        Whenever the model calls the `density_map` attribute, provide only the
        unmasked pixels for inference.
        '''
        return self._density_map_masked

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
        return self._pixel_vectors_masked
    
    @property
    def pixel_vectors_xyz(self) -> list[NDArray[np.float64]]:
        return self._pixel_vectors_xyz_masked

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
        adjusted depending on whether or not the user is fitting a monopole by
        passing ell = [0, ...] at instantiation.

        :param Theta: Parameter samples from the prior distribtuion, as
            generated by the Nested Sampler.
        '''
        multipole_terms = self.model(Theta)

        if self.monopole_is_fitted:
            return self.poisson_log_likelihood(
                rate_parameter=multipole_terms,
                density_map=self.density_map
            )
        else:
            return self.point_by_point_log_likelihood(
                multipole_signal=multipole_terms,
                density_map=self.density_map
            )
    
    def model(self, Theta: NDArray[np.float64]) -> NDArray[np.float64]:
        '''
        The essential idea of to iterate over each ell, computing the multipole
        signal for that particular order. For example, if a user specifies
        `ells = [1,2,3]`, we compute the dipole signal, then the quadrupole
        signal, then the octupole signal, summing them cumulatively.
        '''
        nlive = Theta.shape[0]
        if self.multipole_orders:
            amplitude_like = np.column_stack([
                self._theta_param(Theta, f'M{ell}')
                for ell in self.multipole_orders
            ])
        else:
            amplitude_like = np.zeros((nlive, 0))
        
        signal = np.ones((self.n_unmasked, nlive))
        for i, ((ell, phi_idxs), (ell, theta_idxs)) in enumerate(
            zip(self.phi_indices.items(), self.theta_indices.items())
        ):
            signal += self.compute_multipole_signal(
                multipole_amplitudes=amplitude_like[:, i],
                multipole_longitudes=Theta[:, phi_idxs],
                multipole_latitudes=Theta[:, theta_idxs]
            )
        
        if self.monopole_is_fitted:
            mean_number_density = self._theta_param(Theta, 'M0')

            if self.rms_map is None:
                return mean_number_density[None, :] * signal
            else:
                rms_slope = self._theta_param(Theta, 'rms_slope')
                rms_ratio = self.rms_map / self.rms_ref
                rms_scaling = vectorised_rms_signal(rms_ratio, rms_slope)
                return mean_number_density[None, :] * rms_scaling * signal
        else:
            return signal
    
    def compute_multipole_signal(self,
            multipole_amplitudes: NDArray[np.float64],
            multipole_longitudes: NDArray[np.float64],
            multipole_latitudes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        '''
        :param multipole_amplitudes: Vector of multipole amplitudes, shape (n_live,).
            For example, for an octupole, the vector would be the n_live samples of
            the octupole amplitude.
        :param multipole_longitudes: Matrix of multipole azimuthal angles,
            shape (n_live, ell). For example, for an octupole, the matrix would have
            three azimuthal (phi) angles for the three octupole unit vectors.
        :param multipole_latitudes: Matrix of multipole polar anglea,
            shape (n_live, ell). For example, for an octupole, the matrix would have
            three polar (theta) angles for the three octupole unit vectors.
        :param pixel_vectors: List of Cartesian coordinates of pixel vectors (of
            form [X, Y, X]). X, Y, and Z are vectors of shape (n_pixels,). 
        :return: Vectorised multipole signal of shape (n_pixels, n_live). For example,
            for an octupole, the output O_{ijk} p_{i} p_{j} p_{k} is determined,
            which is the inner product of the octupole tensor and pixel unit
            vectors.
        '''
        ell = multipole_longitudes.shape[1]

        if ell == 1:
            dipole_signal = compute_dipole_signal(
                dipole_amplitude=multipole_amplitudes,
                dipole_longitude=multipole_longitudes.squeeze(), # remove length 1 axes
                dipole_colatitude=multipole_latitudes.squeeze(),
                pixel_vectors=self.pixel_vectors # reshape to (n_pix, 3)
            )
            return dipole_signal

        elif ell == 2:
            cartesian_quadrupole_vectors = vectorised_spherical_to_cartesian(
                phi_like=multipole_longitudes,
                theta_like=multipole_latitudes
            )
            quadrupole_tensor = vectorised_quadrupole_tensor(
                amplitude_like=multipole_amplitudes,
                cartesian_quadrupole_vectors=cartesian_quadrupole_vectors
            )
            quadrupole_signal = multipole_pixel_product_vectorised(
                multipole_tensors=quadrupole_tensor,
                pixel_vectors=self.pixel_vectors_xyz,
                ell=2
            )
            return quadrupole_signal

        else:
            cartesian_multipole_vectors = vectorised_spherical_to_cartesian(
                phi_like=multipole_longitudes,
                theta_like=multipole_latitudes
            )
            multipole_tensor = multipole_tensor_vectorised(
                amplitude_like=multipole_amplitudes,
                cartesian_multipole_vectors=cartesian_multipole_vectors
            )
            multipole_signal = multipole_pixel_product_vectorised(
                multipole_tensors=multipole_tensor,
                pixel_vectors=self.pixel_vectors_xyz,
                ell=multipole_longitudes.shape[-1]
            )
        return multipole_signal

    def _cache_parameter_indices(self) -> None:
        self._parameter_indices = {
            name: self.prior.index_for(name)
            for name in self.prior.parameter_names
        }

    def _theta_param(self, Theta: NDArray[np.float64], name: str) -> NDArray[np.float64]:
        idx = self._parameter_indices[name]
        return Theta[:, idx]
