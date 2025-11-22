import numpy as np
import healpy as hp
import pymaster as nmt
from typing import Optional, Tuple, Dict, Any, List


class PowerSpectrumPipeline:

    def __init__(
        self,
        nside: int = 64,
        mask: Optional[np.ndarray] = None,
        iter_nmt: int = 5,
        bins_input: Optional[np.ndarray] = None,
    ) -> None:
        
        """
        A full end-to-end pipeline for angular power-spectrum analysis using 
            HEALPix and NaMaster.

        This class provides tools for:

        - Applying masks to input maps.
        - Constructing density-contrast (δ) maps.
        - Computing masked and unmasked power spectra using `hp.anafast`.
        - Computing NaMaster coupled and decoupled pseudo-Cl spectra.
        - Convolving theoretical/model power spectra through NaMaster workspaces.
        - Constructing jackknife masks for spatial resampling.
        - Computing jackknife means and standard deviations with interpolation over 
            invalid multipoles.

        :param nside: HEALPix resolution parameter of the input maps.
        :param mask: Binary mask array where 1 indicates unmasked (visible) 
                    pixels and 0 indicates masked pixels.
        :param iter_nmt: Number of NaMaster purification iterations used when 
                        constructing `NmtField` objects.
        :param bins_input: Array of multipole bin edges. If None, bin edges are 
                            generated automatically.
        """

    
        self.nside = nside
        self.mask = mask
        self.iter = iter_nmt
        self.bins_input = bins_input
        self.lmax = 3 * nside - 1

    @staticmethod
    def map_to_density_contrast(map_in: np.ndarray,
                                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the density contrast δ = (ρ - mean) / mean.

        :param map_in: Input scalar HEALPix map.
        :param mask: Optional mask array where 0 = masked.

        
        :return: Density contrast map.
        """
        if mask is not None:
            m = hp.ma(map_in)
            m.mask = (mask == 0)
            mean_val = np.mean(m.compressed())
            delta = hp.ma((map_in - mean_val) / mean_val)
            delta.mask = (mask == 0)
            return delta.filled()

        mean_val = np.mean(map_in)
        return (map_in - mean_val) / mean_val

    
    def process_input_map(self, full_map: np.ndarray) -> Dict[str, Any]:
        """
        Process an input map by applying the mask, computing density contrasts,
        and computing masked & unmasked anafast angular power spectra.

        :param full_map: Input HEALPix map.

        :return: Dictionary containing:
            - masked_map
            - delta
            - delta_masked
            - cl
            - cl_masked
        """
        masked = hp.ma(full_map)
        if self.mask is not None:
            masked.mask = (self.mask == 0)

        delta = self.map_to_density_contrast(full_map, mask=None)
        delta_masked = self.map_to_density_contrast(masked.filled(), mask=self.mask)

        cl = hp.anafast(delta, lmax=self.lmax)
        cl_masked = hp.anafast(delta_masked, lmax=self.lmax)

        return {
            "masked_map": masked,
            "delta": delta,
            "delta_masked": delta_masked,
            "cl": cl,
            "cl_masked": cl_masked,
        }

    def compute_power(
        self,
        map_in: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[nmt.NmtWorkspace, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
               np.ndarray]:
        """
        Compute coupled and decoupled NaMaster auto-spectra.

        :param map_in: Input map (usually density contrast).
        :param mask: Mask to apply during spectrum computation (0 = masked).

        :return: Tuple of:
            (workspace, leff, pcl, cl_dec, cl_dec_pure, leff_pure)
        """

        if self.bins_input is not None:
            bins_use = nmt.NmtBin.from_edges(
                self.bins_input[:-1], self.bins_input[1:]
            )
        else:
            log_space = np.logspace(
                np.log10(15),
                np.log10(192),
                15
            )
            bins_high = np.round(log_space).astype(int)

            for i in range(1, len(bins_high)):
                if bins_high[i] <= bins_high[i - 1]:
                    bins_high[i] = bins_high[i - 1] + 1

            bins_raw = (
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13] +
                list(np.clip(bins_high, 15, 192).tolist())
            )

            bins_use = nmt.NmtBin.from_edges(bins_raw[:-1], bins_raw[1:])

        leff = bins_use.get_effective_ells()

        if mask is None:
            mask_use = np.ones(hp.nside2npix(self.nside))
        else:
            mask_use = mask

        f = nmt.NmtField(mask_use, [map_in], n_iter=self.iter)

        pcl = nmt.compute_coupled_cell(f, f)

        w = nmt.NmtWorkspace.from_fields(f, f, bins_use)
        cl_dec = w.decouple_cell(pcl)[0]

        neg = np.where(cl_dec < 0)[0]
        cl_dec[neg] = np.nan

        cl_pure = np.delete(cl_dec, neg)
        leff_pure = np.delete(leff, neg)

        return w, leff, pcl, cl_dec, cl_pure, leff_pure

    @staticmethod
    def convolve_spectrum(cl_in: np.ndarray,
                          workspace: nmt.NmtWorkspace) -> np.ndarray:
        """
        Convolve a model input spectrum using a NaMaster workspace.

        :param cl_in: Input Cl array.
        :param workspace: NaMaster workspace constructed from fields.

        :return: Decoupled, convolved angular power spectrum.
        """
        cl_reshaped = cl_in[None, :]
        coupled = workspace.couple_cell(cl_reshaped)
        decoupled = workspace.decouple_cell(coupled)[0]

        neg = np.where(decoupled < 0)[0]
        decoupled[neg] = np.nan

        return decoupled

    def generate_jackknife_masks(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Produce jackknife masks by subdividing the sky into pixel regions.

        :param mask: Full-resolution binary mask (1 = unmasked, 0 = masked).

        :return: List of jackknife masks.
        """
        nside_high = hp.npix2nside(len(mask))
        nside_low = self.nside//8  # Instead of fixed 64

        mask_low = hp.ud_grade(mask, nside_low, power=-2)

        jk_masks = [mask]
        good_pixels = np.where(mask_low == np.max(mask_low))[0]

        for p in good_pixels:
            # Convert to nest
            p_nest = hp.ring2nest(nside_low, p)

            children_nest = np.arange(self.nside * p_nest,
                                      self.nside * (p_nest + 1))
            children_ring = hp.nest2ring(nside_high, children_nest)

            new_mask = np.copy(mask)
            new_mask[children_ring] = 0
            jk_masks.append(new_mask)

        return jk_masks

    def jackknife_statistics(
        self,
        map_in: np.ndarray,
        jk_masks: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute NaMaster decoupled power spectra for all jackknife masks.

        :param map_in: Input density contrast map.
        :param jk_masks: List of jackknife masks.

        :return: Array of shape (N_jackknife, N_ell) of decoupled spectra.
        """
        spectra = []
        for m in jk_masks:
            _, _, _, cl_dec, _, _ = self.compute_power(map_in, mask=m)
            spectra.append(cl_dec)
        return np.array(spectra)

    @staticmethod
    def interpolate_nan(
        leff: np.ndarray,
        cl_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the mean and standard deviation of jackknife spectra over 
        NaN multipoles.

        :param leff: Effective multipole values of the binned spectrum.
        :param cl_values: Jackknife Cl array with shape (N_jack, N_ell).

        :return: Tuple (mean_interp, std_interp)
        """
        mean = np.nanmean(cl_values, axis=0)
        std = np.nanstd(cl_values, axis=0)

        mean_interp = np.interp(leff, leff[~np.isnan(mean)], mean[~np.isnan(mean)])
        std_interp = np.interp(leff, leff[~np.isnan(std)], std[~np.isnan(std)])

        return mean_interp, std_interp
