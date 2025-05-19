from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.plotting import MapPlotter
from dipoleska.models.dipole import Dipole
import healpy as hp

# load in map_1 with briggs=1 and 'AA' configuration
loader = MapLoader(1, 'AA')
density_map = loader.load(1)

# downscale from nside=512 to nside=64 to speed up inference
density_map = hp.ud_grade(density_map, power=-2, nside_out=64)

# plot the density map
plotter = MapPlotter(density_map)
*_, = plotter.plot_density_map()

# instantiate a dipole model and specify a Poisson likelihood function
model = Dipole(density_map, likelihood='poisson')
model.prior.plot_priors()

# results are by default saved in ultranest_logs/
model.run_nested_sampling()
model.corner_plot()
model.posterior_predictive_check()