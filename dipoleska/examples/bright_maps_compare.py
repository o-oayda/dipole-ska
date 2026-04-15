import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def load_results(results_path: Path) -> dict[str, Any]:
    with results_path.open() as handle:
        return json.load(handle)


def discover_pairs(root: Path) -> list[tuple[str, str, str, Path, Path]]:
    pairs: list[tuple[str, str, str, Path, Path]] = []

    for map_type_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for likelihood_dir in sorted(path for path in map_type_dir.iterdir() if path.is_dir()):
            dipole_dir = likelihood_dir / 'ells-01'
            dipole_quad_dir = likelihood_dir / 'ell-012'
            if not dipole_dir.exists() or not dipole_quad_dir.exists():
                continue

            dipole_results = {
                path.parent.parent.name: path
                for path in dipole_dir.glob('*/info/results.json')
            }
            dipole_quad_results = {
                path.parent.parent.name: path
                for path in dipole_quad_dir.glob('*/info/results.json')
            }

            for dataset in sorted(dipole_results.keys() & dipole_quad_results.keys()):
                pairs.append(
                    (
                        map_type_dir.name,
                        likelihood_dir.name,
                        dataset,
                        dipole_results[dataset],
                        dipole_quad_results[dataset],
                    )
                )

    return pairs


def compute_row(
    map_type: str,
    likelihood: str,
    dataset: str,
    dipole_path: Path,
    dipole_quad_path: Path,
) -> dict[str, Any]:
    dipole_results = load_results(dipole_path)
    dipole_quad_results = load_results(dipole_quad_path)

    logz_01 = float(dipole_results['logz'])
    logz_012 = float(dipole_quad_results['logz'])
    logzerr_01 = float(dipole_results.get('logzerr', math.nan))
    logzerr_012 = float(dipole_quad_results.get('logzerr', math.nan))

    log_bayes_factor_01_vs_012 = logz_01 - logz_012
    log_bayes_factor_err = math.sqrt(logzerr_01**2 + logzerr_012**2)

    return {
        'map_type': map_type,
        'likelihood': likelihood,
        'dataset': dataset,
        'logz_ells01': logz_01,
        'logzerr_ells01': logzerr_01,
        'logz_ells012': logz_012,
        'logzerr_ells012': logzerr_012,
        'log_bayes_factor_ells01_vs_ells012': log_bayes_factor_01_vs_012,
        'log_bayes_factor_err': log_bayes_factor_err,
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        'map_type',
        'likelihood',
        'dataset',
        'logz_ells01',
        'logzerr_ells01',
        'logz_ells012',
        'logzerr_ells012',
        'log_bayes_factor_ells01_vs_ells012',
        'log_bayes_factor_err',
    ]

    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Compare bright_maps evidences by computing the Bayes factor '
            'for dipole (ells-01) versus dipole+quadrupole (ell-012).'
        )
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('bright_maps_30-03-26'),
        help='Root directory containing the bright_maps outputs.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('bright_maps_30-03-26/bright_maps_bayes_factors.csv'),
        help='CSV path for the summary table.',
    )
    args = parser.parse_args()

    pairs = discover_pairs(args.root)
    rows = [
        compute_row(map_type, likelihood, dataset, dipole_path, dipole_quad_path)
        for map_type, likelihood, dataset, dipole_path, dipole_quad_path in pairs
    ]
    rows.sort(
        key=lambda row: row['log_bayes_factor_ells01_vs_ells012'],
        reverse=True,
    )
    write_csv(rows, args.output)

    print(f'Wrote {len(rows)} comparisons to {args.output}')
    for row in rows:
        print(
            f"{row['map_type']},{row['likelihood']},{row['dataset']},"
            f"logB={row['log_bayes_factor_ells01_vs_ells012']:.6f}"
        )


if __name__ == '__main__':
    main()
