# %%
from dipoleska.utils.catalogue_read import CatalogueLoader
import healpy as hp
import matplotlib.pyplot as plt
#%%
# Cosmology Catalogues
catalog = CatalogueLoader(True, 'SFG', 64)
hp.mollview(catalog.get_catalogue())
plt.show()
#%%
# Complete Catalogues
catalog = CatalogueLoader(False, 'SFG', 64, is_SNR=True, minimum_flux=1e-4,minimum_snr=15)
hp.mollview(catalog.get_catalogue())
