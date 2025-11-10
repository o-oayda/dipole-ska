from typing import Any, List, Literal
import healpy as hp
from numpy.typing import NDArray
import numpy as np


class MapLoader:
    def __init__(self,
            briggs_weighting: Literal[-1, 0, 1],
            configuration: Literal['AA', 'AA4']
    ):
        '''
        Class for loading in SKA maps. Call the load method to return a map.

        :param briggs_weighting: Briggs weighting used to generate the map.
        :param configuration: SKA telescope configuration used to 
                            generate the map.
        '''
        if briggs_weighting == -1:
            self.briggs_weighting = 'n1'
        else:
            self.briggs_weighting = str(briggs_weighting)
        
        self.configuration = configuration

    def load(self,
            map_number: int
    ) -> NDArray[np.int_]:
        '''
        Load and return SKA fits density map based on congigured settings
        and specified map number.

        :param map_number: Number appearing in fits file, e.g. `map_1.fits`
            refers to map 1.
        :return: Healpy density map.
        '''
        self.map_number = map_number
        self.file_path = (
            f'data/ska/briggs_{self.briggs_weighting}/{self.configuration}'
        )
        self.file_name = f'map_{self.map_number}.fits'
        try:
            print(f'Reading in {self.file_path}/{self.file_name}...')
            self.density_map = hp.read_map(
                f'{self.file_path}/{self.file_name}',
                nest=False
            )
            return self.density_map
        except FileNotFoundError as e:
            raise Exception(f'''
                            Cannot find file. File details are as follows:
                            Briggs weighting: {self.briggs_weighting}
                            Configuration: {self.configuration}
                            Map number: {self.map_number}
                            Path: {self.file_path}/{self.file_name}
                            ''') from e

class MapCollectionLoader:
    def __init__(self,
                 snr_cut: Literal[5,10],
                 lower_flux_limit: Literal['5e-5', '1e-4', '5e-4', '1e-3'],
                 lower_z_limit: Literal['0', '0.5'],
                 gal_cut: Literal[0,5,10],
                 map_types: List[Literal['counts','rms','alpha','redshift', 
                                         'flux','info']],
                 nside: int = 64
                 ) -> None:
        '''
        Class for loading the updated SKA simulations (10.11.25).
        
        :param snr_cut: SNR cut to apply when loading maps.
        :param lower_flux_limit: Lower flux limit to apply when loading maps.
        :param lower_z_limit: Lower redshift limit to apply when loading maps.
        :param gal_cut: Galactic cut to apply when loading maps.
        :param nside: Healpy nside parameter for the maps.
        '''
        
        self._map_collections: dict[str, Any] = {}
        self.snr_cut = snr_cut
        self.lower_flux_limit = float(lower_flux_limit)
        self.lower_z_limit = float(lower_z_limit)
        self.gal_cut = gal_cut
        self.nside = nside
        self.upper_z_limit = float("5.0")
        self.path_to_files = 'data/ska/mapcollections/'
        self.map_types = map_types
        self.file_configuration = (
            f'_nside{self.nside}_flux{self.lower_flux_limit}_snr{self.snr_cut}'
            f'_z{self.lower_z_limit}_z{self.upper_z_limit}_gal{self.gal_cut}'
        )
        self.map_dict = {
                        'counts': ('countmap', '.fits'),
                        'rms': ('rmsmap', '.fits'),
                        'alpha': ('alphamap', '.fits'),
                        'redshift': ('zhist', '.txt'),
                        'flux': ('fluxhist', '.txt'),
                        'info': ('xa', '.txt')
                        }

    @property
    def map_collections(self
                        ) -> dict[str, Any]:
        
        '''
        Load and return the specified SKA map collections based on the
        configured settings.
        
        :return: Dictionary of loaded maps.
        '''
        
    
        if self._map_collections:
            return self._map_collections

        for map_type in self.map_types:
            if map_type not in self.map_dict:
                raise ValueError(f"Unknown map type: {map_type}")

            base_name, ext = self.map_dict[map_type]
            file_path = (f"{self.path_to_files}{base_name}"
                        f"{self.file_configuration}{ext}")

            try:
                print(f"Reading in {file_path}...")
                if ext == ".fits":
                    data = hp.read_map(file_path, nest=False)
                else:
                    data = np.loadtxt(file_path)
                self._map_collections[map_type] = data
            except Exception as e:
                raise Exception(
                    f"Cannot read file for map type '{map_type}'."
                    f" Path: {file_path}"
                ) from e

        return self._map_collections