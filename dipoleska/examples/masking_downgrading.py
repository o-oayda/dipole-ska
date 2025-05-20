# %%
from dipoleska.utils.map_read import MapLoader
import healpy as hp
from dipoleska.utils.plotting import MapPlotter
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_process import MapProcessor
import matplotlib.pyplot as plt
import numpy as np

#%%
density_map = MapLoader(1, 'AA').load(1)

plotter = MapPlotter(density_map)
plotter.plot_density_map()
plt.show()
# plotter.plot_smooth_map()

#%%
density_map_downscaled = MapProcessor(density_map).change_map_resolution(64)
masked_map_generator = MapProcessor(density_map_downscaled)
masked_chnged_map = masked_map_generator.mask(
    ['north_equatorial','south_equatorial','galactic_plane'],[10,85,10],'G')
plotter = MapPlotter(masked_chnged_map)
plotter.plot_density_map()
plotter.plot_smooth_map()
plt.show()