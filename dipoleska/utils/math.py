from numpy.typing import NDArray
import numpy as np
import healpy as hp

def uniform_to_uniform_transform(
        uniform_deviates: NDArray[np.float64],
        minimum: float,
        maximum: float 
) -> NDArray[np.float64]:
    '''
    Transform uniform deviates on [0, 1] to uniform deviates on
    [minimum, maximum].

    :param uniform_deviates: Array of uniform deviates, of shape (n_deviates,).
    :param minimum: Minimum of the target distribution.
    :param maximum: Maximum of the target distribution.
    :return: Transformed deviates, shape (n_deviates,).
    '''
    return (minimum - maximum) * uniform_deviates + maximum

def uniform_to_polar_transform(
        uniform_deviates: NDArray[np.float64],
        minimum: float = 0.,
        maximum: float = np.pi
    ) -> NDArray[np.float64]:
    '''
    In spherical coordinates, we need to account for the area element
    changing with polar angle when sampling uniformly.
    This function allows one to transform a uniform deviate on [0,1]
    to a 'polar' distribution on [minimum, maximum] where this is taken care of.
    - theta ~ acos(u * (cos maximum - cos minimum) + cos minimum)
    - theta in [minimum, maximum], subdomain of [0, pi]

    :param uniform_deviates: Uniform deviate between 0 and 1.
    :param minimum: Minimum value to bound polar angles between.
    :param maximum: Maximum value to bound polar angles between.
    :return: uniform deviate on latitudinal (polar) angles
    '''
    unif_theta = np.arccos(
        np.cos(minimum) + uniform_deviates * (np.cos(maximum) - np.cos(minimum))
    )
    return unif_theta

def compute_dipole_signal(
        dipole_amplitude: NDArray[np.float64],
        dipole_longitude: NDArray[np.float64],
        dipole_colatitude: NDArray[np.float64],
        pixel_vectors: NDArray[np.float64]
) -> NDArray[np.float64]:
        '''
        For a vectorised call of the dipole model, compute the term D cos(theta),
        where theta is the angle between the dipole vector and a given pixel.
        In other words, compute the pure l=1 spherical harmonic.

        :param dipole_amplitude: Vector of dipole amplitudes, shape (n_live,).
        :param dipole_longitude: Vector of dipole longitudes, shape (n_live,).
        :param dipole_colatitude: Vector of dipole colatitudes, shape (n_live,).
        :param pixel_vectors: Matrix of pixel vectors, of shape (n_pix, 3).
        :return: Dipole spherical harmonic, of shape (n_pix, n_live).
        '''
        dipole_vector = (
              dipole_amplitude[:, None]
            * hp.ang2vec(dipole_colatitude, dipole_longitude)
        )
        dipole_signal = np.einsum('ki,ji->jk', dipole_vector, pixel_vectors)
        return dipole_signal

def sigma_to_prob2D(sigma: list[float]) -> NDArray[np.float64]:
    '''
    Convert sigma significance to mass enclosed inside a 2D normal
    distribution using the explicit formula for a 2D normal.
    
    :param sigma: The levels of significance, e.g. `sigma = [1., 2.]`.
    :returns: The probability enclosed within each significance level.
    '''
    return 1.0 - np.exp(-0.5 * np.asarray(sigma)**2)