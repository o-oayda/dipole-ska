# %%
from dipoleska.utils.modulated_map_generator import ModulatedMapGenerator
from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.plotting import MapPlotter
from dipoleska.models.dipole import Dipole
# %%
density_map = MapLoader(1, 'AA').load(1)
plotter = MapPlotter(density_map)
plotter.plot_density_map()
# %%
modulator = ModulatedMapGenerator(
    density_map,
    dipole_amplitude=0.05,
    dipole_longitude=264,
    dipole_latitude=48
)
dipole_map = modulator.modulated_map(scaling_factor=100)
processor = MapProcessor(dipole_map)
processor.change_map_resolution(nside_out=64)
density_map_dscaled = processor.density_map
# %%
model = Dipole(density_map_dscaled)
model.run_nested_sampling()
model.corner_plot(coordinates=['equatorial'])