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
        :param dipole_longitude: Longitude of the dipole, using the native
            coordinates of the input map. Input in degrees.
        :param dipole_latitude: Latitude of the dipole, using the native
            coordinates of the input map. Input in degrees latitude, running
            from -90 to 90.
        '''
        self.density_map = density_map
        self.dipole_amplitude = dipole_amplitude
        self.dipole_longitude = dipole_longitude
        self.dipole_latitude = dipole_latitude

    def dipole_map(self, nside: int) -> NDArray[np.float64]:
        '''
        Calcuate the dipolar modulation map for an NSIDE, and the dipole parameters.
        
        :param nside: NSIDE for the dipole map.
        
        :return: An all-sky map of dipolar modulation.
        '''
        pixel_vectors = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
        dipole = self.dipole_amplitude * hp.ang2vec(
            self.dipole_longitude,
            self.dipole_latitude,
            lonlat=True
        )
        stacked_vectors = np.vstack(pixel_vectors).T
        dipole_map = 1 + np.dot(stacked_vectors, dipole)
        return dipole_map
    
    def modulated_map(self,
            scaling_factor: float = 100.,
    ) -> NDArray[np.int_]:
        '''
        Generate a dipole modulated sky from an SKA map.
        
        :param scaling factor: The factor by which alm's have to be scaled before
            modulating the map. Used to ensure we have large enough numbers that 
            we don't lose the dipole in the in the poisson draw.
        
        :return: Dipole modulated Healpy density map.
        '''
        # NOTE: The SKA power spectrum is truncated beyond ell = 512, which is
        # also the nside of the map. Therefore, one needs to truncate the power
        # spectrum when calling map2alm by specifying lmax, otherwise there may
        # be cells with negative rate paramaters due to higher order harmonics
        # beyond ell = 512.
        self.scaling_factor = scaling_factor
        self.nside = hp.npix2nside(len(self.density_map))
        self.lmax = self.nside
        assert self.nside == 512, f'''Map of nside {self.nside} used.
Please use the native nside=512 maps instead.'''
        
        self.alm = hp.map2alm(self.density_map, lmax=self.lmax)
        self.scaled_alm = self.scaling_factor * self.alm
        self.reconstructed_map = hp.alm2map(
            self.scaled_alm,
            nside=self.nside,
            lmax=self.lmax
        )
        
        self.dipole_modulation = self.dipole_map(self.nside)
        self.dipole_added_map = self.reconstructed_map * self.dipole_modulation
        self.lambdas = self.dipole_added_map / self.scaling_factor
        final_map = np.random.poisson(self.lambdas)
        
        return final_map