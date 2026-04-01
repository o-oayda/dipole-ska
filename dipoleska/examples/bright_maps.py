from typing import Any
from dipoleska.models.multipole import Multipole
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapCollectionLoader
from dipoleska.utils.plotting import MapPlotter
import matplotlib.pyplot as plt
import os
import healpy as hp
import argparse


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--map_type',
        choices=[
            'FluxInitial_ModifiedSizes_bright', 'FluxFinal_ModifiedSizes_bright'
        ],
        help='Choose a map dir to process.'
    )
    argparser.add_argument(
        '--do_map_comparison',
        action='store_true',
        help='Make a gridded map comparison plot.'
    )
    argparser.add_argument(
        '--run_fits',
        action='store_true',
        help='Run dipole model over all matches.'
    )
    argparser.add_argument(
        '--likelihood',
        choices=['general_poisson', 'general_poisson_rms'],
        default='general_poisson',
        help='Choose a likelihood function for the dipole model.'
    )
    argparser.add_argument(
        '-i',
        '--ells',
        nargs='+',
        type=int,
        default=[0,1]
    )
    args = argparser.parse_args()

    DPI = 100
    MAP_TYPE = args.map_type
    MAKE_COMPARISON = args.do_map_comparison
    RUN_FITS = args.run_fits
    LIKELIHOOD = args.likelihood
    ELLS = args.ells
    ells_str = 'ell-'
    for i in ELLS:
        ells_str += str(i)
    RUN_OUT_DIR = f'bright_maps_30-03-26/{MAP_TYPE}/{LIKELIHOOD}/{ells_str}/'
    os.makedirs(RUN_OUT_DIR, exist_ok=True)

    loader = MapCollectionLoader(base_dirs=[f'data/ska/bright_maps_30-03-26/{MAP_TYPE}'])
    map_dicts = loader.map_collections
    assert type(map_dicts) is list

    n_maps = len(map_dicts)
    all_dmaps = [map_dicts[i]['files']['counts']['data'] for i in range(n_maps)]
    all_rmsmaps = [map_dicts[i]['files']['rms']['data'] for i in range(n_maps)]
    all_fluxcuts = [map_dicts[i]['attrs']['flux'] for i in range(n_maps)]
    all_zcuts = [map_dicts[i]['attrs']['z'] for i in range(n_maps)]
    all_snr_cuts = [map_dicts[i]['attrs']['snr'] for i in range(n_maps)]
    all_gal_cuts = [str(map_dicts[i]['attrs']['gal']) for i in range(n_maps)]
    all_titles = [map_dicts[i]['id'] for i in range(n_maps)]

    # title_dmap_pairs = list(zip(all_titles, all_dmaps))
    # title_dmap_pairs.sort(key=lambda x: x[0])
    # all_titles = [pair[0] for pair in title_dmap_pairs]
    # all_dmaps = [pair[1] for pair in title_dmap_pairs]

    processor = MapProcessor(all_dmaps)
    processor.mask(output_frame='C', load_from_file='gal10_ps')
    all_masked_dmaps = processor.density_maps

    processor = MapProcessor(all_rmsmaps)
    processor.mask(output_frame='C', load_from_file='gal10_ps')
    all_masked_rmsmaps = processor.density_maps

    if MAKE_COMPARISON:
        plotter = MapPlotter(all_masked_dmaps)
        plotter.plot_density_map(
            projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']},
            all_titles=all_titles
        )
        plt.savefig(f'{RUN_OUT_DIR}/all_density_maps_{MAP_TYPE}.png', dpi=DPI, bbox_inches='tight')
        plt.close()

        plotter.plot_smooth_map(
            projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']},
            all_titles=all_titles
        )

        plt.savefig(f'{RUN_OUT_DIR}/all_smooth_maps_{MAP_TYPE}.png', dpi=DPI, bbox_inches='tight')
        plt.close()

    if RUN_FITS:
        for i in range(n_maps):
            run_name = all_titles[i]
            run_name = run_name.split('bright_maps_30-03-26-no_doppler-')[-1]
            masked_dmap = all_masked_dmaps[i]
            masked_rmsmap = all_masked_rmsmaps[i]
            flux_cut = all_fluxcuts[i]
            snr_cut = all_snr_cuts[i]
            save_dir = f'{RUN_OUT_DIR}/{run_name}'

            print(f'Running {run_name}...')

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            processor = MapProcessor(masked_dmap)
            masked_dmap = processor.density_map

            plotter = MapPlotter([masked_dmap, masked_rmsmap])

            plotter.plot_density_map(projview_kwargs={'coord': ['C', 'G']})
            plt.savefig(f'{save_dir}/dmap_rms.png', dpi=300)

            plotter.plot_smooth_map(projview_kwargs={'coord': ['C', 'G']})
            plt.savefig(f'{save_dir}/smooth_dmap_rms.png', dpi=300)

            model = Multipole(
                masked_dmap, 
                likelihood=LIKELIHOOD,
                rms_map=masked_rmsmap if 'rms' in LIKELIHOOD else None,
                ells=ELLS
            )
            model.prior.change_prior(prior_index=-2, new_prior=['Uniform', -1., 1.])
            model.run_nested_sampling(output_dir=RUN_OUT_DIR, run_name=run_name)

            model.corner_plot(
                backend='getdist', coordinates=['equatorial', 'galactic'],
                save_path=f'{save_dir}/corner_{run_name}.pdf'
            )
            model.sky_direction_posterior(
                coordinates=['equatorial', 'galactic'],
                save_path=f'{save_dir}/sky_{run_name}.pdf'
            )
            plt.close()
