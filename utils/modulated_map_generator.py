from typing import Literal
import healpy as hp
from numpy.typing import NDArray
import numpy as np
from utils.map_read import MapLoader


class ModulatedMapGenerator:
    def __init__(self,
                 briggs_weighting: Literal[-1, 0, 1],
                configuration: Literal['AA', 'AA4'],
                dipole_amplitude: float,
                dipole_longitude: float,
                dipole_latitude: float,
    ):
        '''
        Class for creating dipole modulated sky maps by sampling from the 
        mean power spectrum of a sepcific configuration of the SKA maps.

        :param briggs_weighting: Briggs weighting used to generate the map.
        :param configuration: SKA telescope configuration used to generate the map.
        :param dipole_amplitude: Amplitude of the modulating.
        :param dipole_longitude: Longitude of the dipole. Input in degrees.
        :param dipole_latitude: Latitude of the dipole. Input in degrees.
        
        '''
        self.briggs_weighting = briggs_weighting
        self.configuration = configuration
        self.dipole_amplitude = dipole_amplitude
        self.dipole_longitude = dipole_longitude
        self.dipole_latitude = dipole_latitude

    def dipole_alm(self,
                   nside: int,
                   ) -> NDArray[np.complex_]:
        '''
        Perform spherical harmonic decomposition of a dipole modulated sky.
        
        :param nside: NSIDE for the dipole map.
        
        :return: array of alm coefficients for a dipole map.
        '''
        vectors = hp.pix2vec(nside,[i for i in range(hp.nside2npix(nside))])
        dipole = self.dipole_amplitude * self.spherical2cartersian(self.dipole_longitude, 
                                                                   self.dipole_latitude)
        dipole_map = 1 + np.dot(dipole, [vectors[0],vectors[1],vectors[2]])
        dipole_alm = hp.map2alm(dipole_map)
        return dipole_alm
    
    
    def modulated_maps(self,
                      map_counts: int,
                      lower_limit: int = 0,
                      upper_limit: int = 200,
                      ) -> NDArray[np.float_]:
        '''
        Generate a set of dipole modulated skies for a particular SKA configuration.
        The maps are not scaled to the correct mean number density right now.
        Someone should add this.
        
        :param map_counts: Number of maps to generate.
        :param lower_limit: Lower limit of the range of maps being used.
        :param upper_limit: Upper limit of the range of maps being used.
        
        For example, if you only want to use the first 100 maps, you can set your lower
        limit to 0 and your upper limit to 100. 
        
        :return: array of dipole modulated all-sky maps.
        '''
        maps_raw_list = [MapLoader(self.briggs_weighting,self.configuration).load(i) 
                         for i in range(lower_limit, upper_limit)]
        maps_cleaned_list = [item for item in maps_raw_list if item is not None]
        nside = hp.npix2nside(len(maps_cleaned_list[0]))
        
        maps_alm = np.array([hp.map2alm(item) for item in maps_cleaned_list])
        mean_alm = np.mean(maps_alm, axis=0)
        
        modulated_maps = np.empty((map_counts, hp.nside2npix(nside)), dtype=np.float_)
        for i in range(map_counts):
            random_phase = np.exp(np.random.uniform(0,2*np.pi,mean_alm.shape[0])*1.0j)
            modulation = np.imag(mean_alm)*random_phase
            new_alm = np.real(mean_alm) + np.real(modulation) + (np.imag(modulation)*1.0j)
            mods = new_alm + self.dipole_alm(nside)
            unscaled_map = hp.alm2map(mods,nside)
            #Please scale the unscaled_map to the correct mean number density.
            #use the scaled_map as argument for the poisson distribution generator.
            modulated_maps[i] = np.random.poisson(unscaled_map)
        return modulated_maps
    
    @staticmethod
    def spherical2cartersian(input_lon: float,
                             input_lat: float
                             ) -> NDArray[np.float_]:
        '''
        Convert directions to cartesian unit vectors.
        
        :param input_lon: Longitude in degrees.
        :param input_lat: Latitude in degrees.
        
        :return: 3D cartesian unit vector.
        '''
        lon,lat = np.deg2rad(input_lon),np.deg2rad(input_lat)
        x = np.cos(lon) * np.cos(lat)
        y = np.sin(lon) * np.cos(lat)
        z = np.sin(lat)
        return np.array([x,y,z])