from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import healpy as hp
from utils.physics import omega_to_theta
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

class MapPlotter:
    def __init__(self, density_map: NDArray[np.int_]) -> None:
        self.density_map = density_map
        self.default_settings = {
            'cbar': True,
            'cb_orientation': 'vertical',
            'phi_convention': 'counterclockwise',
            'badcol':'white',
            'fontsize':{'title':'18','xtick_label':'15','ytick_label':'15'},
        }

    def plot_density_map(self,
                        cmap: str = 'viridis',
                        cmap_alpha: float=1.0,
                        projview_kwargs: dict = {},
        ) ->  tuple[Axes, Figure]:
        '''
        Plot map smoothed with a moving average.
        
        :param cmap: Colormap to use for the plot.
            Default is 'viridis'.
        :param cmap_alpha: Transparency of the colormap.
            0.0 is fully transparent, 1.0 is fully opaque.
        :param projview_kwargs: Keyword arguments to pass to `hp.projview`.
        
        :return: Tuple of the axes and figure objects.
        '''
        projview_dict = {}
        for key in self.default_settings:
            if key not in projview_kwargs:
                projview_dict[key] = self.default_settings[key]
        for key in projview_kwargs:
                projview_dict[key] = projview_kwargs[key]
        
        if 'title' not in projview_dict:
            projview_dict['title'] = 'Density map'
            
        hp.projview(self.density_map,
                    cmap=MapPlotter.cmap_scaled(cmap, cmap_alpha),
                    **projview_dict,
                    override_plot_properties={'vertical_tick_rotation':True 
                                              if projview_dict.get('cb_orientation')=='vertical' 
                                              else False,'cbar_shrink':0.5
                                              if projview_dict.get('cb_orientation')=='vertical' 
                                              else 0.5},
                    )
        return plt.gca(),plt.gcf()
    
    def plot_smooth_map(self,
                        cmap: str = 'viridis',
                        cmap_alpha: float=1.0,
                        moving_average_kwargs: dict = {},
                        projview_kwargs: dict = {},
        ) ->  tuple[Axes, Figure]:
        '''
        Plot map smoothed with a moving average.
        
        :param cmap: Colormap to use for the plot.
            Default is 'viridis'.
        :param cmap_alpha: Transparency of the colormap.
            0.0 is fully transparent, 1.0 is fully opaque.
        :param moving_average_kwargs: Keyword arguments to pass to
            `MapPlotter.moving_average_smooth`.
        :param projview_kwargs: Keyword arguments to pass to `hp.projview`.
        
        :return: Tuple of the axes and figure objects.
        '''
        self.smoothed_map = self.moving_average_smooth(
            self.density_map,
            **moving_average_kwargs
        )
        
        projview_dict = {}
        for key in self.default_settings:
            if key not in projview_kwargs:
                projview_dict[key] = self.default_settings[key]
        for key in projview_kwargs:
                projview_dict[key] = projview_kwargs[key]
        
        if 'title' not in projview_dict:
            projview_dict['title'] = 'Density map smoothed with moving average over'+str(moving_average_kwargs.get('angle_scale'))+' steradians'
            
        hp.projview(self.smoothed_map,
                    cmap=MapPlotter.cmap_scaled(cmap, cmap_alpha),
                    **projview_dict,
                    override_plot_properties={'vertical_tick_rotation':True 
                                              if projview_dict.get('cb_orientation')=='vertical' 
                                              else False,'cbar_shrink':0.5
                                              if projview_dict.get('cb_orientation')=='vertical' 
                                              else 0.5},
                    )
        return plt.gca(),plt.gcf()

    @staticmethod
    def moving_average_smooth(
            density_map: NDArray[np.int_],
            weights: NDArray[np.float64] | None = None,
            angle_scale: float = 1.
    ) -> NDArray[np.float64]:
        '''
        Smooth the healpy map using a moving average defined over a certain
        angular scale.

        :param density_map: Healpy map to smooth. Assumes that masked pixels
            are filled with np.nan.
        :param weights: Vector of weights of the same shape as density map.
            Can be specified to weight each pixel differently in the moving
            average, for example if one wants to weight by a selection function.
        :param angle_scale: Angle in steradians over which to apply the moving
            average.
            
        :return: Smoothed healpy map.
        '''
        included_pixels = np.where(~np.isnan(density_map))[0]
        smoothed_map = np.empty_like(density_map,dtype=np.float64)
        smoothed_map.fill(np.nan)
        nside = hp.get_nside(density_map)
        smoothing_radius = omega_to_theta(angle_scale)
        
        # if no weights are provided, weight all pixels equally
        if weights is None:
            weights = np.ones_like(density_map).astype('float')
        
        # FIXME: this is fast for nside=64 maps but slow for nside=512
        # this needs to be sped up somehow
        for pixel_index in included_pixels:
            pixel_vector = hp.pix2vec(nside, pixel_index)
            disc = hp.query_disc(nside, pixel_vector, smoothing_radius)
            smoothed_map[pixel_index] = np.nanmean(density_map[disc] * weights[disc])

        return smoothed_map

    @staticmethod
    def cmap_scaled(
                    cmap: str,
                    alpha: float,
                    ) -> ListedColormap:
        '''
        Enhance the transparency of a colormap.
        
        :param cmap: Colormap to enhance.
        :param alpha: Transparency of the colormap.
            0.0 is fully transparent, 1.0 is fully opaque.
            
        :return: Enhanced colormap.
        '''
        cmap = plt.get_cmap(cmap)
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:,0:3] *= alpha
        my_cmap[:,0:3] += np.array([1,1,1])*(1-alpha)
        
        my_cmap = ListedColormap(my_cmap)
        return my_cmap