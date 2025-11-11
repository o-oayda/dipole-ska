from __future__ import annotations

import json
from pathlib import Path

import corner
import healpy as hp
import matplotlib
import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dipoleska.models.multipole import Multipole

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MAP_PATH = ARTIFACT_DIR / "fiducial_map_nside16.npy"
METADATA_PATH = ARTIFACT_DIR / "fiducial_map_nside16.json"
CORNER_PATH = ARTIFACT_DIR / "multipole_corner.png"


class TestMultipoleFiducialIntegration:
    '''
    The idea here is to load a reference simulated map (fiducial map)
    with known multipole parameters and do a quick NS run to verify we recover
    the correct multipole parameters within a 3 sigma tolerance.
    We just check the multipole amplitudes and the dipole angle parameters
    for a fiducial sample with ells=[1,2].
    '''
    @pytest.mark.slow
    def test_fiducial_map_recovery(self, tmp_path: Path) -> None:
        start = time.perf_counter()
        assert MAP_PATH.exists(), "Missing fiducial map artifact."
        assert METADATA_PATH.exists(), "Missing fiducial metadata artifact."

        density_map = np.load(MAP_PATH)
        metadata = json.loads(METADATA_PATH.read_text())
        truths = metadata['parameters']
        nside = metadata['nside']
        ells = metadata.get('ells', [1, 2])

        assert density_map.shape[0] == hp.nside2npix(nside)
        assert np.sum(density_map) >= 1_500_000

        model = Multipole(density_map=density_map, ells=[0] + ells)
        log_dir = tmp_path / "ultranest_logs"
        model.run_nested_sampling(
            step=True,
            reactive_sampler_kwargs={
                "log_dir": str(log_dir),
                "resume": "overwrite"
            },
            run_kwargs={
                "min_num_live_points": 200,
                "min_ess": 200,
                "show_status": True
            }
        )

        samples = model.samples
        parameter_names = model.parameter_names
        assert samples.shape[1] == len(parameter_names)

        name_to_index = {name: idx for idx, name in enumerate(parameter_names)}

        def assert_within_three_sigma(param_name: str) -> None:
            idx = name_to_index[param_name]
            series = samples[:, idx]
            sigma = np.std(series, ddof=0)
            assert sigma > 0
            delta = np.abs(np.mean(series) - truths[param_name])
            assert delta <= 3 * sigma

        assert_within_three_sigma('M1')
        assert_within_three_sigma('M2')

        def angular_assert(param_name: str) -> None:
            idx = name_to_index[param_name]
            series = samples[:, idx]
            truth = truths[param_name]
            deltas = (series - truth + np.pi) % (2 * np.pi) - np.pi
            sigma = np.std(deltas, ddof=0)
            assert sigma > 0
            assert np.abs(np.mean(deltas)) <= 3 * sigma

        angular_assert('phi_l1_0')
        angular_assert('theta_l1_0')

        CORNER_PATH.parent.mkdir(parents=True, exist_ok=True)
        truth_vector = [truths.get(name, np.nan) for name in parameter_names]
        fig = corner.corner(
            samples,
            labels=parameter_names,
            truths=truth_vector,
            show_titles=True
        )
        fig.savefig(CORNER_PATH, dpi=200)
        plt.close(fig)
        assert CORNER_PATH.exists()
        duration = time.perf_counter() - start
        print(f"Multipole fiducial integration completed in {duration:.1f}s")
