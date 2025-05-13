import numpy as np
from numpy.typing import NDArray
import healpy as hp
from typing import Literal

class ProcessMap:
    def __init__(self, density_map: NDArray[np.int_]):
        self.density_map = density_map

    def mask(self) -> None:
        pass

    def change_map_resolution(self,
            method: Literal['upscale', 'downscale'],
            **ud_grade_kwargs
    ) -> None:
        pass