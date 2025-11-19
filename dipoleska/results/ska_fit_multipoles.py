import os
from typing import Any, cast
from dipoleska.models.priors import Prior
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapCollectionLoader
from dipoleska.utils.plotting import MapPlotter
from dipoleska.models.multipole import Multipole
import matplotlib.pyplot as plt
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit multipoles to SKA maps."
    )
    # parser.add_argument(
    #     '--newsizes',
    #     action='store_true'
    # )
    parser.add_argument(
        '--likelihood',
        choices=['general_poisson', 'general_poisson_rms'],
        default='general_poisson'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable all plotting so the script can run in headless environments.'
    )
    parser.add_argument(
        '--ells',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help='Multipole orders to fit (only used for the multipole model).'
    )
    return parser


def build_run_name(args: argparse.Namespace) -> str:
    return (
        f"s{args.snr_cut}_flux{args.lower_flux}_z{args.lower_z}"
        f"_gal{args.gal_cut}_nside{args.nside}_ells{'-'.join(map(str, args.ells))}"
    )


def main() -> None:
    OUTPUT_DIR = 'dipoleska/results/runs'

    parser = build_parser()
    args = parser.parse_args()
    plot_enabled = not args.no_plots

    loader = MapCollectionLoader(use_base_rms=True)
    loader.load(filter_attrs={'newsizes': True})
    data = loader.map_collections
    data = cast(list[dict[str, Any]], data)

    for collection in data:
        dmap = collection['files']['counts']['data']
        rmsmap = collection['files']['rms']['data']
        COLLECTION_ID = collection['id']
        LIKELIHOOD = args.likelihood
        run_name = f'{COLLECTION_ID}_{LIKELIHOOD}_ells{'-'.join(map(str, args.ells))}'
        RUN_DIR = os.path.join(OUTPUT_DIR, run_name)
        os.makedirs(RUN_DIR, exist_ok=True)

        processor = MapProcessor([dmap, rmsmap])
        if 'gal10.0' in COLLECTION_ID:
            mask = 'gal10_ps'
        elif 'gal5.0' in COLLECTION_ID:
            mask = 'gal5_ps'
        else:
            raise Exception('Not sure which mask to use yet.')
        processor.mask(output_frame='C', load_from_file=mask)
        masked_dmap, masked_rmsmap = processor.density_maps

        if plot_enabled:
            plotter = MapPlotter(masked_dmap)
            _, fig = plotter.plot_density_map(
                projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']}
            )
            fig.savefig(os.path.join(RUN_DIR, 'dmap.pdf'), bbox_inches='tight')
            _, fig = plotter.plot_smooth_map(
                projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']}
            )
            fig.savefig(os.path.join(RUN_DIR, 'smoothed_dmap.pdf'), bbox_inches='tight')

            plt.close()

        if 1 in args.ells:
            prior = Prior(choose_prior={'M1': ['Uniform', 0., 0.05]})
        else:
            prior = None

        model = Multipole(
            masked_dmap,
            ells=args.ells,
            prior=prior,
            likelihood=LIKELIHOOD,
            rms_map=masked_rmsmap
        )
        step = False
        for ell in args.ells:
            if ell > 1:
                step = True
                break

        model.run_nested_sampling(
            step=step,
            run_name=run_name, 
            output_dir=OUTPUT_DIR
        )

        if plot_enabled:
            model.corner_plot(
                backend='getdist', coordinates=['equatorial', 'galactic'],
                save_path=os.path.join(RUN_DIR, 'corner.pdf')
            )
            model.sky_direction_posterior(
                coordinates=['equatorial', 'galactic'],
                save_path=os.path.join(RUN_DIR, 'sky_proj.pdf'),
                contour_levels=[1, 2]
            )
            plt.close()


if __name__ == '__main__':
     main()
