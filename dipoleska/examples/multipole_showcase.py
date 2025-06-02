from dipoleska.utils.map_read import MapLoader
from dipoleska.models.multipole import Multipole
from dipoleska.utils.map_process import MapProcessor
import matplotlib.pyplot as plt

# load in map_1 with briggs=1 and 'AA' configuration
loader = MapLoader(1, 'AA')
density_map = loader.load(1)

# downscale from nside=512 to nside=64 to speed up inference
processor = MapProcessor(density_map)
processor.change_map_resolution(nside_out=64)
density_map = processor.density_map

model = Multipole(density_map, ells=[1, 2])
model.prior.plot_priors()

model.run_nested_sampling(step=True)
model.corner_plot()
model.posterior_predictive_check()
model.sky_direction_posterior()
plt.show()