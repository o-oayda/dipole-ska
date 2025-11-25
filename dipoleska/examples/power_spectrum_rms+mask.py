#%%
from dipoleska.utils.map_read import MapCollectionLoader
from dipoleska.models.dipole import Dipole
from dipoleska.utils.map_process import MapProcessor
from dipoleska.utils.map_read import MapCollectionLoader
from dipoleska.powerspectrum.posterior_power_spectrum import PosteriorPowerSpectrum
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


gal_list = [10.0]
path_list = [f"data/ska/masks/mask_gal{int(gal)}+_cel+_ps.fits" 
        for gal in gal_list]

mask = [hp.read_map(path) for path in path_list]
use_mask = mask[0]

loader = MapCollectionLoader()


loader.load(filter_attrs={"nside":64, "newsizes": True, "z":0.5, "snr":10,
                        "flux":1e-5, "gal":gal_list[0]}, 
                        map_types=["counts","rms"])

maps = loader.map_collections
map_to_use = maps[0]['files']['counts']['data']
rms_to_use = maps[0]['files']['rms']['data']

#%%

processor = MapProcessor(map_to_use)
processor.mask(output_frame='C', load_from_file='gal10_ps')
masked_dmap = processor.density_map

rmsmap = rms_to_use
processor = MapProcessor(rmsmap)
processor.mask(output_frame='C', load_from_file='gal10_ps')
masked_rmsmap = processor.density_map

likelihood='general_poisson_rms'
model = Dipole(masked_dmap, likelihood=likelihood, rms_map=masked_rmsmap)
model.run_nested_sampling()

#%%

likelihood='general_poisson'
model2 = Dipole(map_to_use, likelihood=likelihood)
samples_with_rms = model.samples
rms_index = 1
samples_without_rms = np.delete(samples_with_rms, rms_index, axis=1)
posteriorps = PosteriorPowerSpectrum(
    sample_chains=samples_without_rms,
    model=model2.model,
    likelihood=likelihood,
    sample_count=500
)

cl_mean, cl_std = posteriorps.power_spectrum_calculator()

#%%
##quick check to show that the PPS is getting the correct density contrast.

def s2c(alpha, delta):
    x = np.sin(delta) * np.cos(alpha)
    y = np.sin(delta) * np.sin(alpha)
    z = np.cos(delta)
    return np.array([x, y, z])
amp, lon, colat = np.mean(model.samples[:,-3:], axis=0)
vec = hp.pix2vec(64,[i for i in range(hp.nside2npix(64))])
dipole_mod = amp * s2c(lon, colat)
base_sample = (1 + np.dot(dipole_mod, [vec[0],vec[1],vec[2]]))

plt.loglog(cl_mean-hp.anafast((base_sample-np.mean(base_sample))/np.mean(
                                                        base_sample),lmax=64))