from dipoleska.models.dipole import Dipole
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapCollectionLoader
from dipoleska.utils.plotting import MapPlotter
import matplotlib.pyplot as plt


RUN_OUT_DIR = 'preview_runs'

loader = MapCollectionLoader(base_dirs=['data/ska/fixed_maps_preview'])
map_dicts = loader.map_collections
n_maps = len(map_dicts)
all_dmaps = [map_dicts[i]['files']['fsmap']['data'] for i in range(n_maps)]
all_fluxcuts = [map_dicts[i]['attrs']['smin'] for i in range(n_maps)]
all_zcuts = [map_dicts[i]['attrs']['zmin'] for i in range(n_maps)]
all_titles = [f'Smin-{all_fluxcuts[i]}_zmin-0p{all_zcuts[i]}' for i in range(n_maps)]

processor = MapProcessor(all_dmaps)
processor.mask(output_frame='C', load_from_file='gal10_ps')
all_masked_dmaps = processor.density_maps

plotter = MapPlotter(all_masked_dmaps)
plotter.plot_density_map(
    projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']},
    all_titles=all_titles
)
plt.savefig(f'{RUN_OUT_DIR}/all_density_maps.png', dpi=300, bbox_inches='tight')
plt.close()

plotter.plot_smooth_map(
    projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']},
    all_titles=all_titles
)

plt.savefig(f'{RUN_OUT_DIR}/all_smooth_maps.png', dpi=300, bbox_inches='tight')
plt.close()

for i in range(n_maps):
    run_name = all_titles[i]
    masked_dmap = all_masked_dmaps[i]
    model = Dipole(masked_dmap, likelihood='general_poisson')
    model.run_nested_sampling(output_dir=RUN_OUT_DIR, run_name=run_name)

    model.corner_plot(
        backend='getdist', coordinates=['equatorial', 'galactic'],
        save_path=f'{RUN_OUT_DIR}/corner_{run_name}.pdf'
    )
    model.sky_direction_posterior(
        coordinates=['equatorial', 'galactic'],
        save_path=f'{RUN_OUT_DIR}/sky_{run_name}.pdf'
    )
    plt.close()
