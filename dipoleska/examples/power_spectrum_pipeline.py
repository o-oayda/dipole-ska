from dipoleska.utils.map_read import MapLoader
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.modulated_map_generator import ModulatedMapGenerator
from dipoleska.powerspectrum.power_spectrum_pipeline import PowerSpectrumPipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import pymaster as nmt
import healpy as hp
import numpy as np

mpl.rcParams['figure.dpi'] = 200
mpl.rc('text', usetex=True)
mpl.rc('font', size=15)

gal_list = [5.0, 10.0]
path_list = [f"data/ska/masks/mask_gal{int(gal)}+_cel+_ps.fits" 
        for gal in gal_list]
mask = [hp.read_map(path) for path in path_list]

use_mask = mask[1]


full_ucl, full_ucl_masked = [],[]
dip_ucl, dip_ucl_masked = [],[]
ska_ucl_decoupled, dip_ucl_decoupled = [],[]

for i in range(101,151):
    ska_map_high = MapLoader(-1, 'AA').load(i)

    processor_ska = MapProcessor(ska_map_high)
    processor_ska.change_map_resolution(nside_out=64)
    ska_map = processor_ska.density_map

    modulator = ModulatedMapGenerator(
        ska_map_high,
        dipole_amplitude=0.005,
        dipole_longitude=167,
        dipole_latitude=-8
    )
    dipole_map_high = modulator.modulated_map(scaling_factor=100)
    processor_dipole = MapProcessor(dipole_map_high)
    processor_dipole.change_map_resolution(nside_out=64)
    dipole_map = processor_dipole.density_map

    # Initialize pipeline
    pipeline = PowerSpectrumPipeline(
        nside=64,
        mask=use_mask,
        iter_nmt=10
    )

    # =====================================================
    # PROCESS INPUT MAPS
    # =====================================================
    ska_data = pipeline.process_input_map(ska_map)
    dip_data = pipeline.process_input_map(dipole_map)

    # Unpack
    ska_delta = ska_data["delta"]
    ska_delta_masked = ska_data["delta_masked"]
    ska_cl = ska_data["cl"]
    ska_cl_masked = ska_data["cl_masked"]

    dip_delta = dip_data["delta"]
    dip_delta_masked = dip_data["delta_masked"]
    dip_cl = dip_data["cl"]
    dip_cl_masked = dip_data["cl_masked"]

    # =====================================================
    # NMT DECOUPLING (masked density maps)
    # =====================================================
    ska_dec = pipeline.compute_power(ska_delta_masked, mask=use_mask)
    dip_dec = pipeline.compute_power(dip_delta_masked, mask=use_mask)

    ska_ws, ska_leff, _, ska_cl_dec, ska_cl_pure, ska_leff_pure = ska_dec
    dip_ws, dip_leff, _, dip_cl_dec, dip_cl_pure, dip_leff_pure = dip_dec

    # =====================================================
    # Convolution of unmasked Cl
    # =====================================================
    ska_unbinned = pipeline.convolve_spectrum(ska_cl, ska_ws)
    dip_unbinned = pipeline.convolve_spectrum(dip_cl, dip_ws)

    # =====================================================
    # JACKKNIFE
    # =====================================================
    jk_masks = pipeline.generate_jackknife_masks(use_mask)

    ska_jk = pipeline.jackknife_statistics(ska_delta_masked, jk_masks)
    dip_jk = pipeline.jackknife_statistics(dip_delta_masked, jk_masks)

    # Interpolate means/std
    ska_jk_mean, ska_jk_std = pipeline.interpolate_nan(ska_leff, ska_jk)
    dip_jk_mean, dip_jk_std = pipeline.interpolate_nan(dip_leff, dip_jk)

    full_ucl.append(ska_cl)
    full_ucl_masked.append(ska_cl_masked)
    dip_ucl.append(dip_cl)
    dip_ucl_masked.append(dip_cl_masked)
    ska_ucl_decoupled.append(ska_jk_mean)
    dip_ucl_decoupled.append(dip_jk_mean)
    

plt.figure(figsize=(12,6))
plt.plot(np.mean(np.array(dip_ucl), axis=0), 'g.-', label='Dipole Map full sky')
plt.plot(np.mean(np.array(dip_ucl_masked), axis=0), 'k.-', label='Dipole Map masked')
plt.plot(dip_leff,np.mean(np.array(dip_ucl_decoupled), axis=0), 'c.-', 
         label='Dipole Map full sky recon')
plt.xlim(1,200)
plt.ylim(1e-7, None)


mean_dip = np.mean(np.array(dip_ucl), axis=0)
std_dip = np.nanstd(np.array(dip_ucl), axis=0)

mean_dip_masked = np.mean(np.array(dip_ucl_masked), axis=0)
std_dip_masked = np.nanstd(np.array(dip_ucl_masked), axis=0)

dip_recon_mean = np.mean(np.array(dip_ucl_decoupled), axis=0)
dip_recon_std  = np.nanstd(np.array(dip_ucl_decoupled), axis=0)

x_full = np.arange(mean_dip.size)

_floor = 1e-30

plt.fill_between(x_full, np.maximum(mean_dip - std_dip, _floor), 
                 mean_dip + std_dip,
                 color='g', alpha=0.2)
plt.fill_between(x_full, np.maximum(mean_dip_masked - std_dip_masked, _floor), 
                 mean_dip_masked + std_dip_masked,
                 color='k', alpha=0.2)

plt.fill_between(dip_leff, np.maximum(dip_recon_mean - dip_recon_std, _floor), 
                 dip_recon_mean + dip_recon_std,
                 color='c', alpha=0.2)

plt.xscale('log')
plt.yscale('log')
plt.title(f"SKA Map: Briggs n1 AA, 50 mocks")
plt.legend()
plt.show()
