# dipole-ska

## Requirements
We are implementing this on `python 3.12`.
To get set up and working:
1. Clone this repo
2. `cd` into it
3. Make a python virtual environment: `python -m venv .venv`.
4. Activate the virtual environment: `source .venv/bin/activate`
5. Install the package using `pip install -e .` The `-e` flag ensures changes made to the source code will be applied on subsequent imports.

> [!NOTE]
> We are using `matplotlib==3.7.5` to avoid the plotting issue where the contours wrap around the coordinates.

## Adding the Data
Place the SKA maps in their corresponding folder in `data/ska/`.
For example, the maps in `SKA_Briggs1m_AA` would go in `data/ska/briggs_1/AA/`.

> [!NOTE]
> The `.placeholder` files in `data/ska/` are there to make sure the empty directories are tracked by git.

## Quickstart
To quckly get set up and running, as well as to see how this library works,
follow the example below. This will load an SKA map and fit a dipole to it.

```python
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
```