import logging
import os
from typing import Any, cast
import ultranest
import sys
from abc import abstractmethod
from numpy.typing import NDArray
import ultranest.stepsampler
import numpy as np
from dipoleska.models.model_helpers import MapModelMixin

class InferenceMixin:
    '''
    Provides methods for running Nested Sampling algorithms.
    Inherited by each model, which need to certain properties and methods.
    '''
    def __init__(self):
        self._disable_ultranest_logging()

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        raise NotImplementedError(
            'Subclass models must define parameter names property.'
        )

    @abstractmethod
    def log_likelihood(self,
            Theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            'Subclass models must define log_likelihood method.'
        )

    @abstractmethod
    def prior_transform(self,
            uniform_deviates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            'Subclass models must define prior_transform method.'
        )

    def run_nested_sampling(self,
            step: bool = False,
            n_steps: int | None = None,
            output_dir: str = 'ultranest_logs',
            run_num: int | None = None,
            run_name: str | None = None,
            reactive_sampler_kwargs: dict = {},
            run_kwargs: dict = {}
        ) -> None:
        '''
        Perform Nested Sampling using ultranest's `ReactiveNestedSampler`.
        Results are saved in the results attribute.

        :param step: Specify whether or not to use the random step method.
        :param n_steps: If the random step method is specified, this is the
            number of steps as used by `SliceSampler`.
        :param output_dir: Base directory for UltraNest outputs. Each run
            creates a `run_<n>` subdirectory here (when `resume='subfolder'`,
            the default) unless ``run_name`` is provided.
            A `dipoleska_run_info.log` file describing the prior configuration
            is also written into that subdirectory.
        :param run_num: Explicit UltraNest run number to use when
            `resume='subfolder'`. If None (default), UltraNest auto-increments.
        :param run_name: Optional custom folder name (relative to
            ``output_dir``). When supplied, UltraNest writes directly into
            ``output_dir/run_name`` with ``resume='overwrite'``.
        :param reactive_sampler_kwargs: Extra keyword arguments forwarded to
            ``ultranest.ReactiveNestedSampler``.
        :param run_kwargs: Keyword arguments forwarded to
            ``ReactiveNestedSampler.run``.
        '''
        log_dir_override = None
        resume_mode = 'subfolder'
        self._last_run_name = None
        self._prior_overrides = {}

        if run_name is not None:
            if not run_name.strip():
                raise ValueError('run_name, if provided, must be non-empty.')
            resume_mode = 'overwrite'
            log_dir_override = os.path.join(output_dir, run_name)
            os.makedirs(log_dir_override, exist_ok=True)
            self._last_run_name = run_name

        self.ultranest_sampler = ultranest.ReactiveNestedSampler(
            param_names=self.parameter_names,
            loglike=self.log_likelihood,
            transform=self.prior_transform,
            **{
                'log_dir': log_dir_override or output_dir,
                'resume': resume_mode,
                'vectorized': True,
                'run_num': run_num if run_name is None else None,
                **reactive_sampler_kwargs
            }
        )

        if step:
            self._switch_to_step_sampling(n_steps)
        
        self.results = self.ultranest_sampler.run(**run_kwargs)
        self.ultranest_sampler.print_results()
        self._write_prior_log()

        # there is an issue with ultranest plotting when the log likelihood is
        # very negative (e.g. for the point-by-point likelihood)
        # this catches the ValueError raised
        try:
            self.ultranest_sampler.plot()
        except ValueError as e:
            print(e)
        
        if self.results is not None:
            self._samples = np.atleast_2d(np.asarray(self.results['samples']))

            ws_dict = cast(dict[str, Any], self.results['weighted_samples'])
            weights = ws_dict.get('weights')
            weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)
            self._weights = weights_array

            weighted_samples = ws_dict.get('points')
            self._weighted_samples = np.atleast_2d(
                np.asarray(weighted_samples, dtype=np.float64)
            )

            self.log_bayesian_evidence = self.results['logz']
        else:
            raise Exception('Ultranest results are undefined.')

    @property
    def samples(self) -> NDArray[np.float64]:
        return self._samples # type: ignore

    def _switch_to_step_sampling(self, n_steps: int | None = None) -> None:
        if n_steps is None:
            n_steps = 2 * len(self.parameter_names)
        
        self.ultranest_sampler.stepsampler = ultranest.stepsampler.SliceSampler( # type: ignore
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

    def _write_prior_log(self) -> None:
        '''
        Write a simple log file inside the current UltraNest run directory
        enumerating the prior configuration used for this run.
        '''
        sampler_logs = getattr(self.ultranest_sampler, 'logs', None)
        if not isinstance(sampler_logs, dict):
            return

        run_dir = sampler_logs.get('run_dir')
        if not run_dir:
            return

        prior = getattr(self, 'prior', None)
        if prior is None or not hasattr(prior, 'prior_dict'):
            return

        prior_dict = prior.prior_dict
        os.makedirs(run_dir, exist_ok=True)
        logfile = os.path.join(run_dir, 'dipoleska_run_info.log')

        formatter = getattr(self, '_format_alias', None)
        def format_alias(alias):
            if callable(formatter):
                return formatter(alias)
            return str(alias)

        lines = [f'{self.__class__.__name__} run metadata', '-' * 40]
        custom_name = getattr(self, '_last_run_name', None)
        if custom_name:
            lines.append(f'Run name: {custom_name}')

        likelihood = getattr(self, 'likelihood', None)
        if likelihood is not None:
            lines.append(f'Likelihood: {likelihood}')
        map_meta = []
        nside = getattr(self, 'nside', None)
        if nside is not None:
            map_meta.append(f'nside={nside}')
        n_unmasked = getattr(self, 'n_unmasked', None)
        if n_unmasked is not None:
            map_meta.append(f'unmasked_pixels={n_unmasked}')
        if map_meta:
            lines.append('Map info: ' + ', '.join(map_meta))
        lines.append('')

        lines.extend(MapModelMixin._prior_configuration_lines(
            model_label=self.__class__.__name__,
            merged=prior_dict,
            overrides=getattr(self, '_prior_overrides', {}),
        ))
        lines.append('')

        with open(logfile, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))
