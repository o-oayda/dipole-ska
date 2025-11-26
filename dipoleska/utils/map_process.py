from typing import Literal, Optional
import healpy as hp
from numpy.typing import NDArray
import numpy as np
import os
from astropy.io import fits


class MapProcessor:
    def __init__(self,
            density_map: NDArray[np.int_] | list[NDArray[np.int_]]
        ):
        '''Class to mask the SKA maps

        :param density_map: The SKA density map—or a list of maps of identical
            shape—to be masked. The source arrays are never modified in place.
        '''
        self._density_maps = (
            density_map if isinstance(density_map, list) else [density_map]
        )
        lengths = {len(m) for m in self._density_maps}
        if len(lengths) != 1:
            raise ValueError('All maps must have the same number of pixels.')

        self._density_map = self._density_maps[0]
        self.nside = hp.npix2nside(len(self._density_map))
        self.masked_map = np.ones(len(self._density_map), dtype=np.int64)
        self.is_masked = False
    
    @property
    def density_map(self) -> NDArray[np.float64]:
        '''
        Convenience accessor returning only the first map (useful when a single
        map was supplied at construction). Returns float64 so masked pixels can
        be set to ``np.nan``.
        '''
        out_map = self._density_map.astype(np.float64) # makes a copy
        boolean_mask = ~self.masked_map.astype(np.bool_)
        out_map[boolean_mask] = np.nan
        return out_map

    @property
    def density_maps(self) -> list[NDArray[np.float64]]:
        '''
        Return every map provided at construction with the current mask applied.
        Each array is copied and promoted to float64 so masked pixels become
        ``np.nan``.
        '''
        return [self._apply_mask(m) for m in self._density_maps]
    
    @property
    def get_mask(self) -> NDArray[np.int_]:
        '''
        Return the current mask as an integer array (1 for unmasked pixels,
        0 for masked pixels).
        '''
        return self.masked_map.copy()

    def _apply_mask(self, map_in: NDArray[np.int_]) -> NDArray[np.float64]:
        out_map = map_in.astype(np.float64)
        boolean_mask = ~self.masked_map.astype(np.bool_)
        out_map[boolean_mask] = np.nan
        return out_map
    
    def reset_mask(self) -> None:
        '''
        Remove a previously-created mask.
        '''
        self.masked_map = np.ones(len(self._density_map), dtype=np.int64)
        self.is_masked = False

    def mask(self,
            output_frame: Literal['C', 'G', 'E'],
            classification: Optional[list[
                Literal[
                    'north_equatorial',
                    'south_equatorial',
                    'galactic_plane'
                ]
            ]] = None,
            radius: Optional[list[float]] = None,
            load_from_file: Optional[Literal['ps', 'gal5_ps', 'gal10_ps']] = None,
    ):
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
        :param load_from_file: Choose a pre-computed mask to load from disk
            based on a file identifier. This can be combined with the options
            above to add to the loaded mask. Choices available:
            - ps: disc masks around bright point sources
            - gal5_ps: as above but also with a 5 deg Galactic plane mask
            - gal10_ps: as above but with 10 deg
            
        :return: None; access masked map with this object's density_map attribute.
        '''
        if (classification is None) and (load_from_file is None):
            print('No mask specified in arguments. Skipping...')
            return

        masked_pixels = []

        if classification:
            assert radius is not None, (
                'Pass a list of radii when using the classification argument.'
            )

            for iterator in range(len(classification)):
                mask = self._mask_construction(
                    classification[iterator],
                    self.nside,
                    radius[iterator],
                    output_frame
                )
                masked_pixels.append(mask)

        if load_from_file:
            file_masked_idxs = self._load_mask(load_from_file)
            masked_pixels.append(file_masked_idxs)
        
        masked_pixels = np.concatenate(masked_pixels)
        masked_pixels = np.unique(masked_pixels)
        self.masked_map[masked_pixels] = 0
        self.is_masked = True
    
    def _mask_construction(self,
            classification: Literal[
                'north_equatorial', 
                'south_equatorial',
                'galactic_plane'
            ],
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
        
        elif classification=='south_equatorial':
            lon,lat = self._coordinate_conversion(0,-90,'C',output_frame)
            pixel = self._queried_cap(lon,lat,radius,nside)
            
        elif classification=='galactic_plane':
            lon_n,lat_n = self._coordinate_conversion(0,90,'G',output_frame)
            pixels_n = self._queried_cap(lon_n,lat_n,90-radius,nside)
            lon_s,lat_s = self._coordinate_conversion(0,-90,'G',output_frame)
            pixels_s = self._queried_cap(lon_s,lat_s,90-radius,nside)
            pixel_set = set([i for i in range(hp.nside2npix(nside))])-set(
                np.concatenate((pixels_n,pixels_s)))
            pixel=np.array(list(pixel_set),dtype='int')
        
        else:
            raise Exception('Mask type not recognised, see docstring of mask method.')
        
        return np.array(list(pixel), dtype='int')

    def _load_mask(
            self, 
            file_key: Literal['ps', 'gal5_ps', 'gal10_ps']
    ) -> NDArray[np.int_]:
        '''
        Load a mask from disk in the data/ska/masks directory.

        :param file_key: Short file identifier, see _FILEPATH_MAP.

        :return: Array of masked pixel indices, i.e., the pixels with a value
            of 0 in the boolean mask loaded from disk.
        '''
        _FILEPATH_MAP = {
            'ps': 'mask_ps.fits',
            'gal5_ps': 'mask_gal5+_cel+_ps.fits',
            'gal10_ps': 'mask_gal10+_cel+_ps.fits'
        }
        _MASK_PATH = 'data/ska/masks/'

        try:
            filename = _FILEPATH_MAP[file_key]
        except KeyError:
            raise KeyError(f'No file associated with key {file_key}.')

        full_path = os.path.join(_MASK_PATH, filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f'Cannot find path {full_path}.')

        # this will be a boolean mask, 0 being masked
        print(f'Loading mask from {full_path}...')
        mask_table = fits.open(full_path)
        mask_2D = mask_table[1].data['T'] # pyright: ignore[reportAttributeAccessIssue]
        mask_1D = mask_2D.flatten()
        boolean_mask = np.asarray(mask_1D, bool)

        masked_pixel_idxs = np.where(~boolean_mask)[0]
        return masked_pixel_idxs
    
    @staticmethod
    def _coordinate_conversion(lon: float,
                            lat: float,
                            input_frame: Literal['C', 'G', 'E'],
                            output_frame: Literal['C', 'G', 'E']
                            ) -> tuple[np.float64, np.float64]:
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
        rotated_theta, rotated_phi = rotator(theta,phi) # type: ignore
        rotated_lon =  np.rad2deg(rotated_phi)
        rotated_lat = 90-np.rad2deg(rotated_theta)
        if rotated_lon < 0:
            rotated_lon += 360
        return rotated_lon, rotated_lat
    
    @staticmethod
    def _queried_cap(
            lon: np.float64,
            lat: np.float64,
            radius: np.float64 | float,
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
        return np.array(
            list(hp.query_disc(nside, vector, radius=np.deg2rad(radius),))
        )

    def change_map_resolution(self,
            nside_out: int,
            scaling_power: float = -2,
            **ud_grade_kwargs
        ):
        '''
        Change the resolution of the map using the specified method.
        
        :param nside_out: The desired NSIDE of the output map.
        :param scaling_power: The power to which the map is scaled. The default is -2.
            refer to the Healpy's ud_grade documentation for more details.
        :param ud_grade_kwargs: Keyword arguments to pass to the `ud_grade` function.
            
        :return: None; access new map with this object's density_map attribute.
        '''
        if nside_out > hp.npix2nside(len(self._density_map)):
            print('Increasing the resolution of the map will create unwanted atrifacts \
                in the power spectra of the output map. Tread this path with caution.')
        
        if self.is_masked:
            print('Change the map resolution before masking. Aborting...')
            return None
        
        self._density_map = hp.ud_grade(
            self._density_map,
            nside_out,
            power=scaling_power,
            **ud_grade_kwargs
        )
        self.nside = nside_out
        self.masked_map = np.ones(len(self._density_map), dtype=np.int64)
