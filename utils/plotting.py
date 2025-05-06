from numpy.typing import NDArray
import numpy as np

class MapPlotter:
    def __init__(self, density_map: NDArray[np.int_]) -> None:
        self.density_map = density_map

    def plot_density_map(self) -> None:
        pass

    def plot_smooth_map(self) -> None:
        pass