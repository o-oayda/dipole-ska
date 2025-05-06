import numpy as np
from numpy.typing import NDArray

class Posterior:
    def __init__(self, equal_weighted_samples: NDArray[np.float64]) -> None:
        self.samples = equal_weighted_samples

    def corner_plot(self):
        pass

    def posterior_predictive_check(self):
        pass

    def sky_direction_posterior(self):
        pass