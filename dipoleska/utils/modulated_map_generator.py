from typing import Literal
import healpy as hp
from numpy.typing import NDArray
import numpy as np
from dipoleska.utils.map_read import MapLoader
import healpy as hp

class ModulatedMapGenerator:
    def __init__(self,
            density_map: NDArray[np.int_],
            dipole_amplitude: float,
            dipole_longitude: float,
            dipole_latitude: float,
    ):
        '''
        Class for injecting dipole modulation into SKA maps.

        :param density_map: Healpy density map onto which modulations are done.
        :param dipole_amplitude: Amplitude of the dipole.
        :param dipole_longitude: Longitude of the dipole. Input in degrees.
        :param dipole_latitude: Latitude of the dipole. Input in degrees.
        '''
        self.density_map = density_map
        self.dipole_amplitude = dipole_amplitude
        self.dipole_longitude = dipole_longitude
        self.dipole_latitude = dipole_latitude

    def dipole_map(self, nside: int) -> NDArray[np.complex_]:
        '''
        Calcuate the dipolar modulation map for an NSIDE, and the dipole parameters.
        
        :param nside: NSIDE for the dipole map.
        
        :return: An all-sky map of dipolar modulation.
        '''
        pixel_vectors = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
        dipole = self.dipole_amplitude * hp.ang2vec(self.dipole_longitude,
                                                    self.dipole_latitude,lonlat=True)
        stacked_vectors = np.vstack(pixel_vectors).T
        dipole_map = 1+np.dot(stacked_vectors,dipole)
        return dipole_map
    
    def modulated_map(self,
            l_max_input: int | None = None,
            scaling_factor: float = 100,
    ) -> NDArray[np.int_]:
        '''
        Generate a dipole modulated sky from an SKA map.
        
        :param l_max_input: The highest multiple to generate during the spherical 
            harmonic decomposition of the map. The default is 3 * nside - 1.
        :param scaling factor: The factor by which alm's have to be scaled before
            modulating the map. Used to ensure we have large enough numbers that 
            we don't lose the dipole in the in the poisson draw The default is 100.
        
        :return: Dipole modulated Healpy density map.
        '''
        nside = hp.npix2nside(len(self.density_map))
        if l_max_input is None:
            l_max = 3 * nside - 1
        else:
            l_max = l_max_input

        # TODO: understand what is happening below with respect to
        # negatives in the power map and needing to downgrade the
        # dipole-modulated map
        self.alm = hp.map2alm(self.density_map)
        self.scaled_alm = scaling_factor * self.alm
        self.reconstructed_map = hp.alm2map(self.scaled_alm, nside=nside)
        self.dipole_modulation = self.dipole_map(nside)
        modulated_map = self.reconstructed_map * self.dipole_modulation
        modulated_map_dscaled = hp.ud_grade(
            modulated_map,
            nside_out=64,
            power=-2
        )   
        self.lambdas = modulated_map_dscaled / scaling_factor
        final_map = np.random.poisson(
            modulated_map_dscaled / scaling_factor
        )
        return final_map