from typing import Literal
import healpy as hp
from numpy.typing import NDArray
import numpy as np

class MapProcessor:
    def __init__(self,
                 density_map: NDArray[np.int_]
                 ):
        '''Class to mask the SKA maps
        
        :param density_map: The SKA density map to be masked.
        '''
        self.density_map = density_map
        self.nside = hp.npix2nside(len(self.density_map))
    
    def mask(self,
                   classification: list[Literal['north_equatorial',
                        'south_equatorial', 'galactic_plane']],
                radius: list[float],
                output_frame: Literal['C', 'G', 'E'],
                ) -> NDArray[np.int_]:
        '''
        Construct a composite mask for a given set of classifications and their
        corresponding radii.
        If the classification is 'galactic_plane', the mask will cover all
        latitudes between -radius and +radius degrees around the galactic
        plane. If the classification is 'north_equatorial' or 'south_equatorial',
        the mask will cover all pixels within the input angular radius of the 
        respective celestial pole.
        
        :param classification: List of classifications of the mask, can be from
            'north_equatorial','south_equatorial', 'galactic_plane'.
        :param nside: NSIDE of the masked map which is to be constructed.
        :param radius: List of query radii in degrees, one for each classification.
        :param output_frame: Output coordinate frame ('C' for celestial,
            'G' for galactic, 'E' for ecliptic).
            
        :return: The masked map, with all the masked pixels set to 0, while the 
            unmasked pixels are set to 1.
        '''
        
        masked_pixels = []
        for iterator in range(len(classification)):
            mask = self._mask_construction(classification[iterator],self.nside,
                                    radius[iterator],output_frame)
            masked_pixels.append(mask)
        masked_pixels = np.concatenate(masked_pixels)
        masked_pixels = np.unique(masked_pixels)
        template_map = np.ones(hp.nside2npix(self.nside))
        template_map[masked_pixels] = 0
        masked_map = template_map*self.density_map
        self.template_map = template_map
        return masked_map
    
    def _mask_construction(self,
                        classification: Literal['north_equatorial', 
                        'south_equatorial', 'galactic_plane'],
                        nside: int,
                        radius: float,
                        output_frame: Literal['C', 'G', 'E']
                        ) -> NDArray[np.int_]:
        '''
        Construct a mask for a given classification and radius.
        If the classification is 'galactic_plane', the mask will cover all
        latitudes between -radius and +radius degrees around the galactic
        plane. If the classification is 'north_equatorial' or 'south_equatorial',
        the mask will cover all pixels within the input angular radius of the 
        respective celestial pole.
        
        :param classification: Classification of the mask 
            ('north_equatorial','south_equatorial', 'galactic_plane').
        :param nside: NSIDE of the map from which the pixels  
            should be queried.
        :param radius: Query radius in degrees.
        :param output_frame: Output coordinate frame ('C' for celestial,
            'G' for galactic, 'E' for ecliptic).
        
        :return: Array of pixel indices within the limits of the mask.
        '''
        if classification=='north_equatorial':
            lon,lat = self._coordinate_conversion(0,90,'C',output_frame)
            pixel = self._queried_cap(lon,lat,radius,nside)
        if classification=='south_equatorial':
            lon,lat = self._coordinate_conversion(0,-90,'C',output_frame)
            pixel = self._queried_cap(lon,lat,radius,nside)
            
        if classification=='galactic_plane':
            lon_n,lat_n = self._coordinate_conversion(0,90,'G',output_frame)
            pixels_n = self._queried_cap(lon_n,lat_n,90-radius,nside)
            lon_s,lat_s = self._coordinate_conversion(0,-90,'G',output_frame)
            pixels_s = self._queried_cap(lon_s,lat_s,90-radius,nside)
            pixel_set = set([i for i in range(hp.nside2npix(nside))])-set(
                np.concatenate((pixels_n,pixels_s)))
            pixel=np.array(list(pixel_set),dtype='int')
        
        return np.array(list(pixel),dtype='int')
    
    @staticmethod
    def _coordinate_conversion(lon: float,
                            lat: float,
                            input_frame: Literal['C', 'G', 'E'],
                            output_frame: Literal['C', 'G', 'E']
                            ) -> tuple[np.float_, np.float_]:
        '''
        Convert angular coordinates from one frame to another. 
        We use the healpy Rotator class to do this.
        
        :param lon: Longitude in degrees.
        :param lat: Latitude in degrees.
        :param input_frame: Input coordinate frame ('C' for celestial,
            'G' for galactic, 'E' for ecliptic).
        :param output_frame: Output coordinate frame ('C' for celestial,
            'G' for galactic, 'E' for ecliptic).
            
        :return: Tuple of converted longitude and latitude in degrees.
        '''
        rotator = hp.Rotator(coord=[input_frame,output_frame])
        theta,phi = np.deg2rad(90-lat),np.deg2rad(lon)
        rotated_theta, rotated_phi = rotator(theta,phi)
        rotated_lon =  np.rad2deg(rotated_phi)
        rotated_lat = 90-np.rad2deg(rotated_theta)
        if rotated_lon < 0:
            rotated_lon += 360
        return rotated_lon, rotated_lat
    
    @staticmethod
    def _queried_cap(lon: float,
                    lat: float,
                radius: float,
                nside: int
                ) -> NDArray[np.int_]:
        '''
        Query all the pixels within a given radius of a point.
        
        :param lon: Longitude of the point in degrees.
        :param lat: Latitude of the point in degrees.
        :param radius: Query radius in degrees.
        :param nside: NSIDE of the map.
        
        :return: Array of pixel indices within the Queried radius.
        '''
        vector = hp.ang2vec(lon,lat, lonlat=True)
        return np.array(list(hp.query_disc(nside, vector, 
                                        radius=np.deg2rad(radius),)))

    def change_map_resolution(self,
                              nside_out: int,
                              scaling_power: float = -2,
                              **ud_grade_kwargs
                              ) -> NDArray[np.float64 | np.int_]:
        '''
        Change the resolution of the map using the specified method.
        
        :param nside_out: The desired NSIDE of the output map.
        :param scaling_power: The power to which the map is scaled. The default is -2.
            refer to the Healpy's ud_grade documentation for more details.
        :param ud_grade_kwargs: Keyword arguments to pass to the `ud_grade` function.
            
        :return: The modified map.
        '''
        if nside_out > hp.npix2nside(len(self.density_map)):
            print('Increasing the resolution of the map will create unwanted atrifacts \
                in the power spectra of the output map. Tread this path with caution.')
        new_map = hp.ud_grade(self.density_map, nside_out, power=scaling_power,
                              **ud_grade_kwargs)
        return new_map