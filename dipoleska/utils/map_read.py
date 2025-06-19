from typing import Literal
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
        :param configuration: SKA telescope configuration used to generate the map.
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
Path: {self.file_path}/{self.file_name}'''
            ) from e