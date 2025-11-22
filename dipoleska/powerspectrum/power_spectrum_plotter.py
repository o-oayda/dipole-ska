from dipoleska.utils.map_read import MapLoader
import healpy as hp
import numpy as np
from typing import Literal, Tuple
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes



class LegacyPowerSpectrumPlotter:
    def __init__(self,
                 min_map_number: int,
                 max_map_number: int,
                 briggs_weighting: Literal[-1, 0, 1],
                configuration: Literal['AA', 'AA4']
                 ):
        '''
        Class for plotting the Power Spectrum of a type of SKA maps. 
        Call the load method to return a map. 
        You can specify a range of maps to be included in the plot.
        
        :param min_map_number: Minimum map number to be included in the plot.
        :param max_map_number: Maximum map number to be included in the plot.
        :param briggs_weighting: Briggs weighting used to generate the map.
        :param configuration: SKA telescope configuration used to generate the map.
        '''
        self.map_min_number = min_map_number
        self.map_max_number = max_map_number
        self.briggs_weighting = briggs_weighting
        self.configuration = configuration

    def plot_power_spectra(self) -> Tuple[Axes, Figure]:
        '''
        Plot the mean power spectrum, with 1 sigma variance shaded region,
        for the specified range of SKA maps. Also plots individual power spectra
        in light gray for reference.
        
        :return: Matplotlib Axes and Figure objects containing the power spectrum plot.
        '''
        mpl.rc('text', usetex=True)
        mpl.rc('font', size=15)
        plt.rcParams['figure.dpi'] = 200

        cl_collection = []
        loader = MapLoader(self.briggs_weighting, self.configuration)
        for i in range(self.map_min_number, self.map_max_number + 1):
            density_map = loader.load(i)
            density_contrast = (density_map - np.mean(density_map)
                                ) / np.mean(density_map)
            cl = hp.anafast(density_contrast, 
                            lmax=hp.npix2nside(len(density_contrast)))
            cl_collection.append(cl)
        cl_mean = np.array(cl_collection).mean(axis=0)
        cl_std = np.std(cl_collection, axis=0)

        ell = np.arange(len(cl_mean))

        fig = plt.figure(figsize=(10, 4))
        for cl in cl_collection:
            plt.plot(ell, cl, color='lightgray', alpha=0.5)
        plt.plot(ell, cl_mean, color='darkcyan', label='Mean $C_l$',
                 zorder=len(cl_collection)+1)
        plt.fill_between(ell, 
                        cl_mean - cl_std, 
                        cl_mean + cl_std,
                        color='darkcyan', alpha=0.3, label='$1\\sigma$ Variance',
                        zorder=len(cl_collection)+1)

        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(1,len(cl_mean))
        plt.ylim(1e-7, cl_mean.max()*10)
        plt.xlabel('$l$')
        plt.ylabel('$C_l$')
        plt.title(
        f"Power Spectrum for {self.map_max_number - self.map_min_number + 1} SKA mocks for "
            f"Briggs {self.briggs_weighting}_{self.configuration} configuration",
            fontdict={'fontsize': 13}
        )
        plt.legend()
        return plt.gca(), fig