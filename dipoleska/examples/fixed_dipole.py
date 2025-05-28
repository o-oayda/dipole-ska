# Fit a dipole to an ska map with fixed coordinates (needs custom prior)

from dipoleska.models.dipole import Dipole
from dipoleska.models.priors import Prior
from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.map_process import MapProcessor
import numpy as np

density_map = MapLoader(1, 'AA').load(1)
procesor = MapProcessor(density_map)
procesor.change_map_resolution(nside_out=64)
density_map_downscaled = procesor.density_map
mean_count = np.mean(density_map_downscaled)

FIXED_D = 0.002
FIXED_PHI = 1
FIXED_THETA = 1

prior = Prior(
    choose_prior={
        'N': ['Uniform', 0.75 * mean_count, 1.25 * mean_count],
        'D': ['Uniform', FIXED_D - 1e-8, FIXED_D + 1e-8],
        'phi': ['Uniform', FIXED_PHI - 1e-8, FIXED_PHI + 1e-8],
        'theta': ['Uniform', FIXED_THETA - 1e-8, FIXED_THETA + 1e-8],
    }
)

model = Dipole(density_map_downscaled, prior=prior, likelihood='poisson')

# check that priors have been defined correctly
model.prior.plot_priors()

# run NS and check corner
model.run_nested_sampling()
model.corner_plot()
print(f'log Z: {model.log_bayesian_evidence}')