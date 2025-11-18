from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import healpy as hp
from dipoleska.utils.physics import omega_to_theta
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from contextlib import contextmanager
from copy import deepcopy
from tqdm import tqdm
import warnings
from typing import Generator, Sequence
import re
from matplotlib.patches import Patch


_PARAMETER_LATEX_MAP: dict[str, str] = {
    'phi': r'\phi\,(\mathrm{rad})',
    'theta': r'\theta\,(\mathrm{rad})',
    'D': r'\mathcal{D}_{\mathrm{EB}}',
    'Nbar': r'\bar{N}_{\mathrm{sources}}',
    'rms_slope': r'x_{\mathrm{RMS}}',
    'gp_dispersion': r'b_{\mathrm{GP}}',
}

ANGLE_LABEL_OVERRIDES: dict[str, dict[str, dict[str, str]]] = {
    'galactic': {
        'phi': {'symbol': r'l', 'unit': r'\,(^{\circ})'},
        'theta': {'symbol': r'b', 'unit': r'\,(^{\circ})'},
    },
    'equatorial': {
        'phi': {'symbol': r'\mathrm{RA}', 'unit': r'\,(^{\circ})'},
        'theta': {'symbol': r'\mathrm{Dec}', 'unit': r'\,(^{\circ})'},
    },
    'ecliptic': {
        'phi': {'symbol': r'\lambda', 'unit': r'\,(^{\circ})'},
        'theta': {'symbol': r'\beta', 'unit': r'\,(^{\circ})'},
    },
}

def apply_angle_label_override(
        base_label: str,
        angle_key: str,
        overrides: dict[str, dict[str, str]]
    ) -> str:
    '''
    Swap the leading angular symbol in ``base_label`` using the provided override
    map (typically pulled from ``ANGLE_LABEL_OVERRIDES``). Keeps any existing
    subscripts, superscripts, or suffixes intact while appending the override's
    unit string at the end.
    '''
    override = overrides.get(angle_key)
    if not override:
        return base_label

    base_symbol = r'\phi' if angle_key == 'phi' else r'\theta'
    if not base_label.startswith(base_symbol):
        return base_label

    rest = base_label[len(base_symbol):]
    rad_suffix = r'\,(\mathrm{rad})'
    if rest.startswith(rad_suffix):
        rest = rest[len(rad_suffix):]

    symbol = override.get('symbol', base_symbol)
    unit = override.get('unit', '')
    return f'{symbol}{rest}{unit}'


def _parameter_latex_label(parameter_name: str) -> str:
    '''
    Map parameter names to LaTeX labels.

    :param parameter_name: Raw parameter name from the sampler.
    :return: LaTeX label to use in plots.
    '''
    if parameter_name in _PARAMETER_LATEX_MAP:
        return _PARAMETER_LATEX_MAP[parameter_name]

    amplitude_match = re.fullmatch(r'M(\d+)', parameter_name)
    if amplitude_match:
        ell = amplitude_match.group(1)
        return rf'\mathcal{{M}}_{{{ell}}}'

    angle_match = re.fullmatch(r'(phi|theta)_l(\d+)_(\d+)', parameter_name)
    if angle_match:
        angle_type, ell, vec_index = angle_match.groups()
        symbol = r'\phi' if angle_type == 'phi' else r'\theta'
        vec_display = int(vec_index)
        return rf'{symbol}_{{\ell={ell}}}^{{({vec_display})}}'

    return parameter_name


def _sanitise_parameter_name(
        name: str,
        index: int,
        seen_names: set[str]
    ) -> str:
    '''
    Ensure GetDist parameter names are unique and free of special characters.

    :param name: Candidate name for the parameter.
    :param index: Parameter index, used when generating fallback names.
    :param seen_names: Set of names already assigned.
    '''
    candidate = ''.join(
        ch if (ch.isalnum() or ch == '_') else '_'
        for ch in name
    )
    if not candidate:
        candidate = f'param_{index}'
    if candidate[0].isdigit():
        candidate = f'p_{candidate}'

    unique_name = candidate
    suffix = 1
    while unique_name in seen_names:
        unique_name = f'{candidate}_{suffix}'
        suffix += 1
    seen_names.add(unique_name)
    return unique_name


def plot_log_log_histogram(
        data: Sequence[float] | NDArray[np.floating],
        bins: int | Sequence[float] = 10,
        color: str = 'cornflowerblue',
        **hist_kwargs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[Patch]]:
    '''
    Plot a histogram with logarithmic scales on both axes, using bins that are
    uniformly spaced in log space.

    :param data: Input array-like of values; non-positive entries are dropped.
    :param bins: Either the number of bins (int) or a sequence of log-uniform
        bin edges to use directly.
    :param color: Color applied to both the filled bars and their outlines.
    :param hist_kwargs: Extra keyword arguments forwarded to ``plt.hist``.
    :return: The ``(counts, bin_edges, patches)`` tuple returned by
        ``plt.hist``.
    '''
    if 'bins' in hist_kwargs:
        raise TypeError('Pass bin specification via the explicit `bins` argument.')
    if 'color' in hist_kwargs:
        raise TypeError('Pass bar color via the explicit `color` argument.')

    values = np.asarray(data, dtype=np.float64)
    positive_mask = values > 0
    if not np.all(positive_mask):
        removed = int(values.size - positive_mask.sum())
        warnings.warn(
            f'Removed {removed} non-positive entries before plotting on log-log axes.',
            RuntimeWarning,
            stacklevel=2
        )
        values = values[positive_mask]

    if values.size == 0:
        raise ValueError('Log-log histogram requires at least one positive value.')

    if isinstance(bins, (int, np.integer)):
        if bins < 1:
            raise ValueError('Number of bins must be a positive integer.')
        edges = np.logspace(
            np.log10(values.min()),
            np.log10(values.max()),
            int(bins) + 1
        )
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if np.any(edges <= 0):
            raise ValueError('Bin edges must be positive for log spacing.')
        log_widths = np.diff(np.log10(edges))
        if not np.allclose(log_widths, log_widths[0]):
            raise ValueError('Provided bin edges are not uniformly spaced in log space.')

    # note: we make two plt.hist calls to get the 'solid edge with alpha' style
    # the first call needs stepfilled with alpha, the second just an edge
    fill_kwargs = dict(hist_kwargs)
    fill_kwargs.setdefault('histtype', 'stepfilled')
    fill_kwargs.setdefault('alpha', 0.3)
    fill_kwargs['color'] = color
    counts, bin_edges, patches = plt.hist(
        values, bins=edges, **fill_kwargs
    )

    edge_kwargs = dict(hist_kwargs)
    edge_kwargs.setdefault('histtype', 'step')
    edge_kwargs['color'] = color
    edge_kwargs['lw'] = 1.5
    plt.hist(
        values, bins=edges, **edge_kwargs
    )

    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log', nonpositive='clip')

    return counts, bin_edges, patches

@contextmanager
def matplotlib_latex(
        font_family: str = 'sans-serif',
        latex_preamble: str | None = r'\usepackage{amsmath}'
    ) -> Generator[None, None, None]:
    '''
    Temporarily enable LaTeX rendering with an optional font family and preamble.

    :param font_family: Matplotlib font family to use while LaTeX is enabled.
    :param latex_preamble: Preamble string for LaTeX rendering. If None, the
        preamble is left unchanged.
    '''
    rc_params = matplotlib.rcParams
    previous = {
        'text.usetex': rc_params.get('text.usetex', False),
        'font.family': deepcopy(rc_params.get('font.family', [])),
    }
    if latex_preamble is not None:
        previous['text.latex.preamble'] = deepcopy(rc_params.get('text.latex.preamble', ''))

    try:
        rc_params['text.usetex'] = True
        if font_family:
            rc_params['font.family'] = font_family
        if latex_preamble is not None:
            rc_params['text.latex.preamble'] = latex_preamble
        yield
    finally:
        for key, value in previous.items():
            rc_params[key] = value

class MapPlotter:
    def __init__(
            self,
            density_map: NDArray[np.int_ | np.float_]
            | Sequence[NDArray[np.int_ | np.float_]]
        ) -> None:
        self.density_maps = self._validate_density_maps(density_map)
        self.density_map = self.density_maps[0]
        self.smoothed_maps: list[NDArray[np.float64]] | None = None
        self.smoothed_map: NDArray[np.float64] | None = None
        self.default_settings = {
            'cbar': True,
            'cb_orientation': 'vertical',
            'phi_convention': 'counterclockwise',
            'badcolor':'white',
            'fontsize':{'title':'18','xtick_label':'15','ytick_label':'15'},
        }

    @staticmethod
    def _validate_density_maps(
            density_map: NDArray[np.int_ | np.float_]
            | Sequence[NDArray[np.int_ | np.float_]]
        ) -> list[NDArray[np.int_ | np.float_]]:
        '''
        Ensure the provided density map(s) are valid NumPy arrays compatible
        with healpy projections.
        '''
        if isinstance(density_map, np.ndarray):
            maps = [density_map]
        elif isinstance(density_map, Sequence) and not isinstance(
            density_map, (str, bytes)
        ):
            if len(density_map) == 0:
                raise ValueError('density_map sequence cannot be empty.')
            maps = list(density_map)
        else:
            raise TypeError(
                'density_map must be a numpy array or a sequence of numpy '
                'arrays.'
            )

        validated: list[NDArray[np.int_ | np.float_]] = []
        for idx, map_array in enumerate(maps):
            if not isinstance(map_array, np.ndarray):
                raise TypeError(
                    f'density_map entry {idx} must be a numpy array; got '
                    f'{type(map_array).__name__}.'
                )
            if map_array.ndim != 1:
                raise ValueError(
                    f'density_map entry {idx} must be a 1D HEALPix map; got '
                    f'shape {map_array.shape}.'
                )
            if not np.issubdtype(map_array.dtype, np.number):
                raise TypeError(
                    f'density_map entry {idx} must have a numeric dtype; got '
                    f'{map_array.dtype!r}.'
                )
            try:
                hp.npix2nside(map_array.size)
            except Exception as exc:
                raise ValueError(
                    'density_map entry {idx} must have a length compatible '
                    f'with a HEALPix map; got {map_array.size}.'
                ) from exc
            validated.append(map_array)
        return validated

    def _build_projview_kwargs(self, projview_kwargs: dict, default_title: str) -> dict:
        projview_dict = {}
        for key in self.default_settings:
            if key not in projview_kwargs:
                projview_dict[key] = self.default_settings[key]
        for key in projview_kwargs:
            projview_dict[key] = projview_kwargs[key]
        
        if 'title' not in projview_dict:
            projview_dict['title'] = default_title
        return projview_dict

    def _plot_maps(
            self,
            maps_to_plot: Sequence[NDArray[np.int_ | np.float_]],
            cmap: str,
            cmap_alpha: float,
            projview_dict: dict,
            title_was_user_supplied: bool
        ) -> tuple[Axes | list[Axes], Figure]:
        n_maps = len(maps_to_plot)
        axes: list[Axes] = []
        figure: Figure | None = None
        figsize = projview_dict.pop('figsize', None)

        if n_maps > 1:
            if 'sub' in projview_dict:
                raise ValueError(
                    "Remove 'sub' from projview_kwargs when plotting multiple "
                    "maps; layout is handled automatically."
                )
            n_cols = int(np.ceil(np.sqrt(n_maps)))
            n_rows = int(np.ceil(n_maps / n_cols))
            # Use an explicit figure sized to fit subplots and keep previous
            # panels intact within this call.
            existing_fig = projview_dict.get('fig')
            if existing_fig is None:
                if figsize is None:
                    figsize = (6 * n_cols, 4 * n_rows)
                figure = plt.figure(figsize=figsize)
                projview_dict['fig'] = figure.number
            elif isinstance(existing_fig, Figure):
                figure = existing_fig
                projview_dict['fig'] = figure.number
            else:
                figure = plt.figure(existing_fig)
        elif figsize is not None and 'fig' not in projview_dict:
            # Single plot with requested size
            figure = plt.figure(figsize=figsize)
            projview_dict['fig'] = figure.number
        else:
            n_rows = n_cols = 1

        base_title = projview_dict.get('title')

        for idx, current_map in enumerate(maps_to_plot):
            map_projview_dict = dict(projview_dict)
            if n_maps > 1:
                map_projview_dict['sub'] = (n_rows, n_cols, idx + 1)
                if base_title and not title_was_user_supplied:
                    map_projview_dict['title'] = f'{base_title} {idx + 1}'

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                hp.projview(
                    current_map,
                    cmap=MapPlotter.cmap_scaled(cmap, cmap_alpha),
                    **map_projview_dict,
                    override_plot_properties={
                        'vertical_tick_rotation': (
                            True
                            if map_projview_dict.get('cb_orientation')
                            == 'vertical'
                            else False
                        ),
                        'cbar_shrink': (
                            0.5
                            if map_projview_dict.get('cb_orientation')
                            == 'vertical'
                            else 0.5
                        )
                    },
                )
                axes.append(plt.gca())
                if figure is None:
                    figure = plt.gcf()

        if n_maps > 1 and figure is not None:
            # Reduce whitespace between subplots for a more compact layout.
            figure.subplots_adjust(wspace=0.06, hspace=0.)

        return (
            axes[0] if len(axes) == 1 else axes,
            figure if figure is not None else plt.gcf()
        )

    def plot_density_map(self,
                        cmap: str = 'viridis',
                        cmap_alpha: float=1.0,
                        projview_kwargs: dict = {},
        ) ->  tuple[Axes | list[Axes], Figure]:
        '''
        Plot a density map (or multiple maps as subplots when provided).
        
        :param cmap: Colormap to use for the plot.
            Default is 'viridis'.
        :param cmap_alpha: Transparency of the colormap.
            0.0 is fully transparent, 1.0 is fully opaque.
        :param projview_kwargs: Keyword arguments to pass to `hp.projview`.
        
        :return: Tuple of the axes (single axes or list when subplots are
            created) and figure objects.
        '''
        projview_dict = self._build_projview_kwargs(projview_kwargs, 'Density map')
        axes, fig = self._plot_maps(
            self.density_maps,
            cmap,
            cmap_alpha,
            projview_dict,
            title_was_user_supplied='title' in projview_kwargs
        )
        return axes, fig
    
    def plot_smooth_map(self,
                        cmap: str = 'viridis',
                        cmap_alpha: float=1.0,
                        moving_average_kwargs: dict = {},
                        projview_kwargs: dict = {},
        ) ->  tuple[Axes | list[Axes], Figure]:
        '''
        Plot map smoothed with a moving average. When multiple maps are
        provided at initialisation, they are plotted as subplots.
        
        :param cmap: Colormap to use for the plot.
            Default is 'viridis'.
        :param cmap_alpha: Transparency of the colormap.
            0.0 is fully transparent, 1.0 is fully opaque.
        :param moving_average_kwargs: Keyword arguments to pass to
            `MapPlotter.moving_average_smooth`.
        :param projview_kwargs: Keyword arguments to pass to `hp.projview`.
        
        :return: Tuple of the axes (single axes or list when subplots are
            created) and figure objects.
        '''
        self.smoothed_maps = [
            self.moving_average_smooth(current_map, **moving_average_kwargs)
            for current_map in self.density_maps
        ]
        self.smoothed_map = self.smoothed_maps[0]
        
        projview_dict = self._build_projview_kwargs(projview_kwargs, 'Smoothed density map')
        axes, fig = self._plot_maps(
            self.smoothed_maps,
            cmap,
            cmap_alpha,
            projview_dict,
            title_was_user_supplied='title' in projview_kwargs
        )
        return axes, fig

    @staticmethod
    def moving_average_smooth(
            density_map: NDArray[np.int_ | np.float_],
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
        for pixel_index in tqdm(included_pixels):
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
