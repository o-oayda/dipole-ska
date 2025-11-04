#%%
from dipoleska.utils.power_spectrum_plotter import PowerSpectrumPlotter
#%%
#Plot the mean power spectrum for 50 SKA maps
ca, cf = PowerSpectrumPlotter(101, 150, -1, 'AA').plot_power_spectra()