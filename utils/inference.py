import logging
import ultranest
import sys

class Inference:
    '''
    Provides methods for running Nested Sampling alghorithms.
    Inherited by each model.
    '''
    def __init__(self):
        self._disable_ultranest_logging()

    def run_nested_sampling(self,
            step: bool = False,
            n_steps: int = None,
            reactive_sampler_kwargs: dict = {},
            run_kwargs: dict = {}
        ) -> None:
        '''
        Perform Nested Sampling using ultranest's `ReactiveNestedSampler`.
        Results are saved in the results attribute.

        :param step: Specify whether or not to use the random step method.
        :param n_steps: If the random step method is specified, this is the
            number of steps as used by `SliceSampler`.
        :param reactive_sampler_kwargs: Keyword arguments for the
            `ReactiveNestedSampler`.
        :param run_kwargs: Keyword arguments for the sampler's run method.
        '''
        self.ultranest_sampler = ultranest.ReactiveNestedSampler(
            param_names=self.parameter_names,
            loglike=self.log_likelihood,
            transform=self.prior_transform,
            **{
                'log_dir': 'ultranest_logs',
                'resume': 'subfolder',
                'vectorized': True,
                **reactive_sampler_kwargs
            }
        )

        if step:
            self._switch_to_step_sampling(n_steps)
        
        self.results = self.ultranest_sampler.run(**run_kwargs)
        self.ultranest_sampler.print_results()

        # there is an issue with ultranest plotting when the log likelihood is
        # very negative (e.g. for the point-by-point likelihood)
        # this catches the ValueError raised
        try:
            self.ultranest_sampler.plot()
        except ValueError as e:
            print(e)
        
        self.samples = self.results['samples']
        self.log_bayesian_evidence = self.results['logz']

    def _switch_to_step_sampling(self, n_steps: int | None = None) -> None:
        if n_steps is None:
            n_steps = 2 * len(self.param_names)

        self.ultranest_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=n_steps,
            generate_direction=(
                ultranest.stepsampler.generate_mixture_random_direction
            )
        )

    def _disable_ultranest_logging(self) -> None:
        '''
        Ultranest spams debug messages by default.
        Running this function should disable them.
        Taken from https://github.com/JohannesBuchner/UltraNest/issues/31.
        '''
        unest_logger = logging.getLogger("ultranest")
        unest_handler = logging.StreamHandler(sys.stdout)
        unest_handler.setLevel(logging.WARN)
        ultranest_formatter = logging.Formatter(
            "%(levelname)s:{}:%(message)s".format("ultranest")
        )
        unest_handler.setFormatter(ultranest_formatter)
        unest_logger.addHandler(unest_handler)
        unest_logger.setLevel(logging.WARN)