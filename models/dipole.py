from numpy.typing import NDArray
import numpy as np

class Likelihood:
    def __init__(self):
        pass

    def point_by_point_log_likelihood():
        pass

    def poisson_log_likelihood():
        pass

class Dipole(Likelihood):
    def __init__(self):
        pass

    def prior_transform(self,
            uniform_deviates: NDArray[np.float64]
        ) -> NDArray[np.float64]:
        pass

    def model(
            Theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass