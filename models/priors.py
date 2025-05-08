from models.default_priors import prior_defaults
from typing import Callable
from utils.math import (
    uniform_to_uniform_transform,
    uniform_to_polar_transform
)
from typing import Dict
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

class Prior:
    def __init__(self,
            prior_summary: Dict[str, list[str | int]] | None = None,
            choose_default: str | None = None
    ):
        '''
        Class for constructing n-dimensional prior distribution functions.
        This class can be passed to a model to specify its sampling distribution.
        The transform functions are stored in the `prior_transforms` attribute.
        
        :param prior_summary: Dictionary specifying the priors for each of
            a model's parameters. For example, for a single-parameter model with
            parameter `theta` drawn from a uniform distribution between 0 and 20,
            specify `prior_summary: {'theta': ['Uniform', 0, 20]}`.
        :param choose_default: Instead of specifying the priors explicitly,
            use default priors for typical models, like a pure dipole.
            These defaults are housed in `models/default_priors.py`.
        :raises: AssertionError if both prior_summary and choose_default
            are None.
        '''
        self.prior_dict = prior_summary
        self.choose_default = choose_default
        
        assert (
            (self.prior_dict is not None)
            or
            (self.choose_default is not None)
        ), "Both prior_summary and choose_default cannot be None."

        if self.prior_dict is not None:
            self._construct_priors()
        else:
            self._load_default_priors()
            self._construct_priors()
    
    def transform(self,
            uniform_deviates: NDArray[np.float64]
        ) -> NDArray[np.float64]:
        '''
        Convert samples on the n-dimensional unit cube to samples in prior
        likelihood space.

        :param uniform_deviates: Samples on the unit cube,
            of shape (n_live, n_dim).
        :return: Deviates transformed to prior distribution,
            of shape (n_live, n_dim).
        '''
        return np.asarray(
            [
                ptform(uniform_deviates[:, i])
                for i, ptform in enumerate(self.prior_transforms)
            ]
        ).T
    
    def change_prior(self,
            prior_index: int,
            new_prior: list[str | int]
        ) -> None:
        '''
        Change the sampling distribution for one of the parameters.

        :param prior_index: The index of the parameter to change in the
            `prior_transforms` list. For example, if the parameters are
            `['D', 'l', 'b']` and one wants to change 'D', then `prior_index=0`.
        :param new_prior: Specify the new sampling distribution. See the
        `__init__` docstring for how to format this distribution.
        '''
        new_distribution = self._get_prior_callable(new_prior[0])
        new_minimum = new_prior[1]
        new_maximum = new_prior[2]
        
        new_callable = self._construct_callable(
            callable_prior_function=new_distribution,
            minimum=new_minimum,
            maximum=new_maximum
        )
        self.prior_transforms[prior_index] = new_callable
    
    def remove_prior(self,
            prior_index: int
    ) -> None:
        '''
        Remove one of the prior transforms as specified by an index.

        :param prior_index: The index of the parameter to change in the
            `prior_transforms` list. For example, if the parameters are
            `['D', 'l', 'b']` and one wants to change 'D', then `prior_index=0`.
        '''
        self.prior_transforms.pop(prior_index)
        self.parameter_names.pop(prior_index)
        self.ndim -= 1

    def plot_priors(self) -> None:
        '''
        Plot the prior distributions by sampling from the prior transforms
        and binning. Useful for checking the priors are actually what the user
        intends.
        '''
        uniform_deviates = np.random.rand(1_000_000)
        _, axs = plt.subplots(
            nrows=1,
            ncols=len(self.prior_transforms),
            layout='constrained'
        )
        
        for i, (ax, ptform) in enumerate(zip(axs, self.prior_transforms)):
            param_samples = ptform(uniform_deviates)
            # sorted_samples = np.sort(param_samples)

            ax.hist(
                param_samples,
                bins=50,
                density=True,
                alpha=0.3,
                color='cornflowerblue'
            )
            ax.hist(
                param_samples,
                bins=50,
                density=True,
                histtype='step',
                color='cornflowerblue',
                linewidth=1.5
            )
            # TODO: add prior pdfs for each prior transform function
            # ax.plot(
            #     sorted_samples,
            #     self.prior_density_functions,
            #     color='tomato',
            #     linewidth=1.5
            # )
            ax.set_title(f'{self.parameter_names[i]}')
        
        plt.show()

    def _construct_priors(self) -> None:
        '''
        Transform the dictionary passed to `Prior` at instantiation into a list
        of callable functions, to be used by this class's `transform` method.
        These functions are stored in the `prior_transforms` attribute.
        '''
        self.parameter_names = list(self.prior_dict.keys())
        self.ndim = len(self.parameter_names)

        self.prior_transforms = []
        for value in self.prior_dict.values():
            distribution = value[0]
            minimum = value[1]
            maximum = value[2]

            callable_prior_function = self._get_prior_callable(distribution)
            constructed_callable = self._construct_callable(
                callable_prior_function,
                minimum,
                maximum
            )
            self.prior_transforms.append(constructed_callable)

    def _load_default_priors(self) -> None:
        '''
        If a user has not specified the priors expicitly but rather chosen
        from a default set, this function loads that set.
        '''
        self.prior_dict = prior_defaults[self.choose_default]
    
    def _get_prior_callable(self,
            distribution: str
    ) -> Callable:
        '''
        Transform a user-specified distribution string into a callable function.

        :param distribution: Distribution string.
        :return: Corresponding callable function.
        '''
        distribution_to_function = {
            'Uniform': uniform_to_uniform_transform,
            'Polar': uniform_to_polar_transform
        }
        return distribution_to_function[distribution]
    
    def _construct_callable(self,
            callable_prior_function: Callable,
            minimum: float,
            maximum: float
    ) -> Callable:
        '''
        Make wrapper function to be used when calling this class's `transform`
        method. This ensures that the prior function is bounded, as determined
        by the user's choices or the default prior set.

        :param callable_prior_function: Function which takes 1D uniform deviate
            on [0,1] and transforms to a different distribution. Must have
            minimum and maximum kwargs, specifying the end points of this
            target distribution.
        :param minimum: Desired minimum of the target distribution.
        :param maximum: Desired maximum of the target distribution.
        '''
        return lambda u, minimum=minimum, maximum=maximum: (
            callable_prior_function(u, minimum, maximum)
        )