from dipoleska.models.default_priors import prior_defaults
from typing import Callable
from dipoleska.utils.math import (
    uniform_to_uniform_transform,
    uniform_to_polar_transform
)
from typing import Dict
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from dipoleska.utils.plotting import _parameter_latex_label


class Prior:
    def __init__(self,
            choose_prior: Dict[str, list[str | float | np.floating]] | str,
    ):
        '''
        Class for constructing n-dimensional prior distribution functions.
        This class can be passed to a model to specify its sampling distribution.
        The transform functions are stored in the `prior_transforms` attribute.
        
        :param choose_prior: Can specify either (1) a dictionary of priors for
            each of a model's parameters or (2) a default prior.
            
            - (1) For example, for a single-parameter model with parameter
                `theta` drawn from a uniform distribution between 0 and 20,
                specify `prior_summary: {'theta': ['Uniform', 0, 20]}`.
            - (2) Instead of specifying the priors explicitly, use default
                priors for typical models, like a pure dipole. These defaults
                are housed in `models/default_priors.py`.
        
        :raises: AssertionError if both prior_summary and choose_default
            are None.
        '''
        if type(choose_prior) is dict:
            self.prior_dict = choose_prior
            self._construct_priors()
        
        elif type(choose_prior) is str:
            self.choose_default = choose_prior
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
            new_prior: list[str | (float | np.float64)] 
        ) -> None:
        '''
        Change the sampling distribution for one of the parameters.

        :param prior_index: The index of the parameter to change in the
            `prior_transforms` list. For example, if the parameters are
            `['D', 'l', 'b']` and one wants to change 'D', then `prior_index=0`.
        :param new_prior: Specify the new sampling distribution. See the
        `__init__` docstring for how to format this distribution.
        '''
        new_callable = self._prior_alias_to_callable(new_prior)
        self.prior_transforms[prior_index] = new_callable
    
    def add_prior(self,
            prior_index: int,
            prior_name: str,
            prior_alias: list[str | (float | np.float64)]
        ) -> None:
        '''
        Insert a new prior definition into the list of transforms.

        :param prior_index: Location in `prior_transforms` and
            `parameter_names` at which to insert the new prior.
        :param prior_name: Name of the parameter associated with the prior.
        :param prior_alias: Sampling distribution alias. Same structure as the
            `new_prior` argument in `change_prior`.
        '''
        assert isinstance(prior_name, str), 'Prior name must be a string.'
        assert 0 <= prior_index <= len(self.prior_transforms), (
            f'Prior index ({prior_index}) out of bounds for '
            f'{len(self.prior_transforms)}-dimensional prior.'
        )
        assert prior_name not in self.parameter_names, (
            f'Prior name "{prior_name}" already exists.'
        )

        new_callable = self._prior_alias_to_callable(prior_alias)
        self.prior_transforms.insert(prior_index, new_callable)
        self.parameter_names.insert(prior_index, prior_name)
        self.ndim += 1
        self._name_to_index = {name: idx for idx, name in enumerate(self.parameter_names)}

        # Maintain `prior_dict` ordering so future operations stay consistent.
        prior_items = list(self.prior_dict.items())
        prior_items.insert(prior_index, (prior_name, prior_alias))
        self.prior_dict = dict(prior_items)
    
    def remove_prior(self,
            prior_indices: list[int]
    ) -> None:
        '''
        Remove the prior transforms as specified by their indices.

        :param prior_indices: The indices of the parameters to change in the
            `prior_transforms` list. For example, if the parameters are
            `['D', 'l', 'b']` and one wants to change 'D', then `prior_indices=[0]`.
        '''
        for prior_index in sorted(prior_indices, reverse=True):
            self.prior_transforms.pop(prior_index)
            self.parameter_names.pop(prior_index)
            self.ndim -= 1
        self._name_to_index = {name: idx for idx, name in enumerate(self.parameter_names)}

    def plot_priors(self) -> None:
        '''
        Plot the prior distributions by sampling from the prior transforms
        and binning. Useful for checking the priors are actually what the user
        intends.
        '''
        uniform_deviates = np.random.rand(1_000_000)
        fig, axs = plt.subplots(
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

            ax.set_yticks([])
            latex_label = _parameter_latex_label(self.parameter_names[i])
            ax.set_title(f'${latex_label}$')

        fig.supylabel('Probability density')
        
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
        self._name_to_index = {name: idx for idx, name in enumerate(self.parameter_names)}
        
        for value in self.prior_dict.values():
            
            distribution = value[0]
            assert type(distribution) is str, (
                f'Distribution ({distribution}) needs to be a string.'
            )
            
            minimum = value[1]
            maximum = value[2]
            assert (
                isinstance(minimum, (float, np.floating))
                and
                isinstance(maximum, (float, np.floating))
            ), (
                f'Bounds on priors ',
                f'(type min: {type(minimum)}, type max: {type(maximum)})'
                'must be floats.'
            )
            
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
    
    def _prior_alias_to_callable(self,
            prior_alias: list[str | (float | np.float64)]
    ) -> Callable:
        '''
        Convert a prior alias (distribution name and bounds) into the callable
        transform used by the sampler.
        '''
        assert len(prior_alias) >= 3, (
            'Prior alias must contain a distribution and two bounds.'
        )
        assert (
            type(prior_alias[0]) is str
        ), 'Prior alias must start with a distribution string.'
        assert (
            isinstance(prior_alias[1], (float, np.floating))
            and
            isinstance(prior_alias[2], (float, np.floating))
        ), 'Prior alias bounds must be floats.'

        distribution_callable = self._get_prior_callable(prior_alias[0])
        return self._construct_callable(
            callable_prior_function=distribution_callable,
            minimum=prior_alias[1],
            maximum=prior_alias[2]
        )

    def index_for(self, name: str) -> int:
        if name not in self._name_to_index:
            raise KeyError(f'Parameter "{name}" not found in prior.')
        return self._name_to_index[name]

    def optional_index_for(self, name: str) -> int | None:
        return self._name_to_index.get(name)
    
    def _construct_callable(self,
            callable_prior_function: Callable,
            minimum: float | np.float64,
            maximum: float | np.float64
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
