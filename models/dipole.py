from numpy.typing import NDArray
import numpy as np
from scipy.stats import poisson

class Likelihood:
    def __init__(self):
        pass

    def point_by_point_log_likelihood(
            dipole_signal: NDArray[np.float64],
            density_map: NDArray[np.int_]
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

    def poisson_log_likelihood(
            rate_parameter: NDArray[np.float64],
            density_map: NDArray[np.int_]
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

class Dipole(Likelihood):
    def __init__(self):
        pass

    def prior_transform(self,
            uniform_deviates: NDArray[np.float64]
        ) -> NDArray[np.float64]:
        pass

    def model(
            Theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass