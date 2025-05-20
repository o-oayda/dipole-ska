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
processor = MapProcessor(density_map)
masked_density_map = processor.mask(
    classification=['north_equatorial','south_equatorial','galactic_plane'],
    radius=[10,85,10],
    output_frame='G'
)
plotter = MapPlotter(masked_density_map)
plotter.plot_density_map()
plt.show()
# plotter.plot_smooth_map()

#%%
changed_map = MapProcessor(masked_density_map).change_map_resolution(64)
masked_map_generator = MapProcessor(changed_map)
masked_chnged_map = masked_map_generator.mask(
    ['north_equatorial','south_equatorial','galactic_plane'],[10,85,10],'G')
plotter = MapPlotter(masked_chnged_map)
plotter.plot_density_map()
plotter.plot_smooth_map()
plt.show()

mask = masked_chnged_map != 0
# print(masked_chnged_map)
plt.figure()
plt.hist(
    masked_chnged_map[mask],
    # bins=np.arange(0, np.max(masked_map_generator.density_map)+10)
    bins = np.linspace(500, 1200, 100)
)

plt.show()