# %%
from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.plotting import MapPlotter
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_process import MapProcessor
import matplotlib.pyplot as plt
#%%
density_map = MapLoader(1, 'AA').load(1)
plotter = MapPlotter(density_map)
plotter.plot_density_map()
plt.show()
#%%
processor = MapProcessor(density_map)
processor.change_map_resolution(nside_out=64)
processor.mask(
    classification=['north_equatorial', 'south_equatorial', 'galactic_plane'],
    radius=[10,85,10],
    output_frame='G'
)
plotter = MapPlotter(processor.density_map)
plotter.plot_density_map()
plotter.plot_smooth_map()
plt.show()