import healpy as hp
import numpy as np
from typing import Callable, Literal
from numpy.typing import NDArray

class PosteriorPowerSpectrum:
    def __init__(self,
               sample_chains: NDArray,
               model: Callable[[NDArray], NDArray],
               likelihood: Literal['point', 'poisson', 'poisson_rms',
                                'general_poisson', 'general_poisson_rms'],
               sample_count: int = 1000
               ):
        '''
        Class for computing the mean power spectrum of the posterior 
        samples generated from Nested Sampling Runs. Note that the input
        model should be constructed from a map of the same nside as the 
        raw ska maps. Look at the relevant example for more details.
        
        :param sample_chains: The posterior samples from the nested sampling
                              run.
        :param model: A callable model that takes in the sample chains and
                      outputs the corresponding sky maps.
        :param likelihood: The likelihood type used in the model.
        :param sample_count: The number of posterior samples to draw for
                             power spectrum estimation.
                             
        :return: A tuple containing the mean power spectrum and its standard
                 deviation.
        '''
        self.sample_chains = sample_chains
        self.model = model
        self.sample_count = sample_count
        self.likelihood = likelihood
        
    def power_spectrum_calculator(self) -> tuple[NDArray, NDArray]:
        n_total = len(self.sample_chains)
        n_select = self.sample_count
        selected_indices = np.random.choice(n_total, size=n_select, 
                                            replace=False)
        selected_samples = self.sample_chains[selected_indices]
        model_samples = self.model(selected_samples)
        if self.likelihood in ['point', 'poisson', 'poisson_rms']:
            posterior_sampled_models = model_samples
        elif self.likelihood in ['general_poisson', 'general_poisson_rms']:
            posterior_sampled_models, _ = model_samples

        cl_collection = []
        for sample_number in range(posterior_sampled_models.shape[1]):
            sample = posterior_sampled_models[:,sample_number]
            density_contrast = (sample - np.mean(sample))/np.mean(sample)
            cl = hp.anafast(density_contrast, 
                            lmax=hp.npix2nside(len(density_contrast)))
            cl_collection.append(cl)
        cl_mean = np.array(cl_collection).mean(axis=0)
        cl_std = np.std(cl_collection, axis=0)
        return cl_mean, cl_std
    
    
        