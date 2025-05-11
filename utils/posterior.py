import numpy as np
from numpy.typing import NDArray
from corner import corner
import matplotlib.pyplot as plt
import os

class Posterior:
    def __init__(self,
            equal_weighted_samples: NDArray[np.float64] | None = None,
            run_number: int | None = None
    ) -> None:
        assert not (
            (equal_weighted_samples is None)
            and
            (run_number is None)
        ), 'Specify either equal weighted samples or a run path, not both.'
        
        self._load_samples_from_log(run_number)

    def _load_samples_from_log(self,
            run_number: int | None = None
    ) -> None:
        if run_number is None:
            pass
        else:
            self.load_path = (
                f'ultranest_logs/run{run_number}/chains/equal_weighted_post.txt'
            )
            assert os.path.exists(
                self.load_path
            ), f'Cannot find path ({self.load_path}).'
            
            self.samples = np.loadtxt(self.load_path, skiprows=1)
            self.parameter_names = np.loadtxt(
                self.load_path,
                max_rows=1,
                dtype=str
            )

    def corner_plot(self, **corner_kwargs):
        corner(
            self.samples,
            **{
                'labels': self.parameter_names,
                'bins': 50,
                'show_titles': True,
                'title_fmt': '.3g',
                'title_quantile': (0.025,0.5,0.975),
                'quantiles': (0.025,0.5,0.975),
                'smooth': 1,
                'smooth1d': 1,
                **corner_kwargs
            }
        )
        plt.show()

    def posterior_predictive_check(self):
        pass

    def sky_direction_posterior(self):
        pass