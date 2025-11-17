import os
from typing import Any, cast
from dipoleska.models.priors import Prior
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapCollectionLoader
from dipoleska.utils.plotting import MapPlotter
from dipoleska.models.multipole import Multipole
import matplotlib.pyplot as plt
import argparse
import numpy as np
import healpy as hp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit multipoles to SKA maps."
    )
    parser.add_argument(
        '--newsizes',
        action='store_true'
    )
    parser.add_argument(
        '--likelihood',
        choices=['general_poisson', 'general_poisson_rms']
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
    parser = build_parser()
    args = parser.parse_args()
    plot_enabled = not args.no_plots

    loader = MapCollectionLoader()
    loader.load(filter_attrs={'newsizes': True})
    data = loader.map_collections
    data = cast(list[dict[str, Any]], data)

    for collection in data:
        dmap = collection['files']['counts']['data']
        rmsmap = collection['files']['rms']['data']
        COLLECTION_ID = collection['id']
        LIKELIHOOD = args.likelihood
        PATH_TO_DMAP = collection['files']['counts']['path']
        subdirs = PATH_TO_DMAP.split('/')
        relative_path_in_data = '/'.join(map(str, subdirs[2:4])) # remove data/ska

        processor = MapProcessor([dmap, rmsmap])
        processor.mask(output_frame='C', load_from_file='gal10_ps')
        masked_dmap, masked_rmsmap = processor.density_maps

        hp.projview(rmsmap)
        plt.show()

        if plot_enabled:
            plotter = MapPlotter(masked_dmap)
            plotter.plot_density_map(
                projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']}
            )
            plotter.plot_smooth_map(
                projview_kwargs={'badcolor': 'grey', 'coord': ['C', 'G']}
            )
            plt.show()

        if 1 in args.ells:
            prior = Prior(choose_prior={'M1': ['Uniform', 0., 0.05]})
        else:
            prior = None

        print(collection)
        model = Multipole(
            masked_dmap,
            ells=args.ells,
            prior=prior,
            likelihood=LIKELIHOOD,
            rms_map=masked_rmsmap
        )
        step = True

        if plot_enabled:
            model.prior.plot_priors()

        run_name = f'{COLLECTION_ID}_{LIKELIHOOD}_ells{'-'.join(map(str, args.ells))}'
        model.run_nested_sampling(
            step=step, run_name=run_name, output_dir=os.path.join('dipoleska/results/runs', relative_path_in_data)
        )
        RUN_DIR = model.get_run_dir()

        if plot_enabled:
            model.corner_plot(
                backend='getdist', coordinates=['equatorial', 'galactic'],
                save_path=RUN_DIR
            )
            model.sky_direction_posterior(
                coordinates=['equatorial', 'galactic'], save_path=RUN_DIR
            )
            plt.show()


if __name__ == '__main__':
     main()
