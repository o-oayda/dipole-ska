from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.plotting import MapPlotter
from dipoleska.models.dipole import Dipole
from dipoleska.utils.map_process import MapProcessor
import matplotlib.pyplot as plt

# load in map_1 with briggs=1 and 'AA' configuration
loader = MapLoader(1, 'AA')
density_map = loader.load(1)

# downscale from nside=512 to nside=64 to speed up inference;
# also mask 10 degrees above and below the Galactic plane
processor = MapProcessor(density_map)
processor.change_map_resolution(nside_out=64)
processor.mask(
    classification=['galactic_plane'],
    radius=[10],
    output_frame='C'
)
density_map = processor.density_map

# plot the density map
plotter = MapPlotter(density_map)
*_, = plotter.plot_density_map()

# instantiate a dipole model and specify a Poisson likelihood function
model = Dipole(density_map, likelihood='poisson')
model.prior.plot_priors()

# results are by default saved in ultranest_logs/
model.run_nested_sampling()
model.corner_plot(coordinates=['equatorial'], backend='getdist')
model.posterior_predictive_check()
model.sky_direction_posterior()
plt.show()
