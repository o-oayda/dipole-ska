
from typing import Literal
import healpy as hp
from numpy.typing import NDArray
import numpy as np
from astropy.io import fits

class CatalogueLoader:
    def __init__(self,
                is_doppler: bool,
                source_type: Literal['AGN', 'SFG'],
                nside: int,
                minimum_flux: float=10**(-6),
                is_SNR: bool=False,
                maximum_flux: float | None=None,
                minimum_snr: float | None=None,
                ):
        '''
        Class for loading the thinner SKA catalogues. Converts the catalogues 
        to a healpy map. Optionally applies a flux cut and SNR cut before binning.

        :param is_doppler: Whether the map is a Doppler boosted catalogue.
        :param source_type: Type of sources listed in the catalogue: AGN or SFG.
        :param nside: nside parameter for the output map.
        :param is_SNR: Whether the catalogue contains SNR or not.
        :param minimum_flux: Minimum flux cut to be applied (in Jy).
        :param maximum_flux: Maximum flux cut to be applied (in Jy).
        :param minimum_snr: Minimum SNR cut to be applied (only if is_SNR is True).
        '''
        global_path = 'data/ska/catalogues_6nov/'
        self.snr = is_SNR
        self.nside = nside
        self.minimum_flux = minimum_flux
        self.maximum_flux = maximum_flux
        self.minimum_snr = minimum_snr
        if self.snr is True:
            self.file_name = global_path+\
            'Randoms-Masked-With-Fluxes-Completeness-Sampled-nogalmask-with-PyBDSF-Comp.fits'
        else:
            path = f'Mock0_{source_type}_cc_full1'
            if is_doppler is True:
                path += '_Doppler'
            self.file_name = global_path+path+'.fits'

    def get_catalogue(self) ->  NDArray[np.int_]:
        
        '''
        Clean the SKA catalogue and convert it to a healpy map.

        :return: Healpy map of the catalogue.
        '''
        try:
            print(f'Reading the catalogue...')
            catalogue = fits.open(self.file_name)[1].data
        except FileNotFoundError as e:
            raise Exception(f'File not found') from e
        
        ra = catalogue['Ra']
        dec = catalogue['Dec']
        alpha = catalogue['alpha']
        
        if self.snr is True:
            flux = catalogue['Flux_final']
            rms = catalogue['RMS_final']
            snr = catalogue['SNR']
            if self.maximum_flux is None:
                self.maximum_flux = np.max(flux)
            valid_flux_indices = np.where((flux>self.minimum_flux) & 
                                          (flux<self.maximum_flux))[0]
            ra, dec = ra[valid_flux_indices], dec[valid_flux_indices]
            flux, alpha = flux[valid_flux_indices], alpha[valid_flux_indices],
            snr, rms = snr[valid_flux_indices], rms[valid_flux_indices]
            
            if self.minimum_snr is not None:
                valid_snr_indices = np.where(snr>self.minimum_snr)[0]
                self.ra, self.dec = ra[valid_snr_indices], dec[valid_snr_indices]
                self.flux, self.alpha = flux[valid_snr_indices], alpha[valid_snr_indices]
                self.snr, self.rms = snr[valid_snr_indices], rms[valid_snr_indices]
            
            else: 
                self.ra, self.dec = ra, dec
                self.flux, self.alpha = flux, alpha
                self.snr, self.rms = snr, rms
            
            
        else:
            flux = catalogue['S(780MHz) [Jy]']
            if self.maximum_flux is None:
                self.maximum_flux = np.max(flux)
            valid_flux_indices = np.where((flux>self.minimum_flux) & 
                                          (flux<self.maximum_flux))[0]
            self.ra, self.dec = ra[valid_flux_indices], dec[valid_flux_indices]
            self.flux, self.alpha = flux[valid_flux_indices], alpha[valid_flux_indices]
        
        hmap = np.zeros(hp.nside2npix(self.nside))
        pixels = hp.ang2pix(self.nside,self.ra,self.dec,lonlat=1)
        self.hmap = hmap + np.bincount(pixels, minlength=hp.nside2npix(self.nside))
        return self.hmap