# %%
from dipoleska.utils.map_read import MapLoader
import healpy as hp
from dipoleska.utils.plotting import MapPlotter
from dipoleska.utils.map_process import MaskedMapGenerator
from dipoleska.utils.map_process import MapResolutionChanger

#%%
density_map = MapLoader(1, 'AA').load(1)

plotter = MapPlotter(density_map)
plotter.plot_density_map()
plotter.plot_smooth_map()

#%%
masked_map_generator = MaskedMapGenerator(density_map)
masked_density_map = masked_map_generator.masked_map(
    ['north_equatorial','south_equatorial','galactic_plane'],[10,85,10],'G')
plotter = MapPlotter(masked_density_map)
plotter.plot_density_map()
plotter.plot_smooth_map()

#%%
changed_map = MapResolutionChanger(masked_density_map).change_map_resolution(64)
masked_map_generator = MaskedMapGenerator(changed_map)
masked_chnged_map = masked_map_generator.masked_map(
    ['north_equatorial','south_equatorial','galactic_plane'],[10,85,10],'G')
plotter = MapPlotter(masked_chnged_map)
plotter.plot_density_map()
plotter.plot_smooth_map()
