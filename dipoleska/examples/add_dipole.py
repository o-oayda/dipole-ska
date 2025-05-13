# %%
from dipoleska.utils.modulated_map_generator import ModulatedMapGenerator
from dipoleska.utils.map_read import MapLoader
import healpy as hp
from dipoleska.utils.plotting import MapPlotter
from dipoleska.models.dipole import Dipole
# %%
density_map = MapLoader(1, 'AA').load(1)
density_map_dscaled = hp.ud_grade(
    density_map,
    nside_out=64,
    power=-2
)

plotter = MapPlotter(density_map_dscaled)
plotter.plot_density_map()
plotter.plot_smooth_map()
# %%
modulator = ModulatedMapGenerator(
    density_map,
    dipole_amplitude=0.05,
    dipole_longitude=264,
    dipole_latitude=48
)
dipole_map = modulator.modulated_map(scaling_factor=100)
# %%
model = Dipole(dipole_map)
model.run_nested_sampling()
model.corner_plot()