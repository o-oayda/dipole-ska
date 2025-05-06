from numpy.typing import NDArray
import numpy as np
import healpy as hp
from utils.physics import omega_to_theta

class MapPlotter:
    def __init__(self, density_map: NDArray[np.int_]) -> None:
        self.density_map = density_map
        self.default_settings = {
            'graticule': True,
            'graticule_labels': True
        }

    def plot_density_map(self, **projview_kwargs) -> None:
        hp.projview(
            self.density_map,
            {
                **self.default_settings,
                **projview_kwargs
            }
        )

    def plot_smooth_map(self,
                moving_average_kwargs: dict = {},
                projview_kwargs: dict = {}
        ) -> None:
        '''
        Plot map smoothed with a moving average.

        :param moving_average_kwargs: Keyword arguments to pass to
            `MapPlotter.moving_average_smooth`.
        :param projview_kwargs: Keyword arguments to pass to `hp.projview`.
        '''
        self.smoothed_map = self.moving_average_smooth(
            self.density_map,
            **moving_average_kwargs
        )
        hp.projview(
            self.smoothed_map,
            {
                **self.default_settings,
                **projview_kwargs
            }
        )

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
        '''
        included_pixels = np.where(~np.isnan(density_map))[0]
        smoothed_map = np.empty_like(density_map)
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