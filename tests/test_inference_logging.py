from types import SimpleNamespace
from pathlib import Path

import numpy as np

from dipoleska.utils.inference import InferenceMixin
from dipoleska.models.priors import Prior


class DummyModel(InferenceMixin):
    def __init__(self) -> None:
        super().__init__()
        self._prior = Prior(choose_prior={'A': ['Uniform', 0.0, 1.0]})
        self._parameter_names = ['A']

    @property
    def parameter_names(self) -> list[str]:
        return self._parameter_names

    @property
    def prior(self) -> Prior:
        return self._prior

    def log_likelihood(self, Theta: np.ndarray) -> np.ndarray:
        return np.zeros(Theta.shape[0])

    def prior_transform(self, uniform_deviates: np.ndarray) -> np.ndarray:
        return uniform_deviates

    @staticmethod
    def _format_alias(alias):
        return str(alias)


def test_inference_writes_prior_log(tmp_path: Path) -> None:
    model = DummyModel()
    run_dir = tmp_path / "run_001"
    model.ultranest_sampler = SimpleNamespace(logs={'run_dir': str(run_dir)})

    model._write_prior_log()

    log_file = run_dir / "dipoleska_prior.log"
    assert log_file.exists()
    contents = log_file.read_text()
    assert 'DummyModel prior configuration' in contents
    assert 'A: [\'Uniform\', 0.0, 1.0]' in contents
