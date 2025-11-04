#%%
from dipoleska.utils.power_spectrum_plotter import PowerSpectrumPlotter
#%%
ca, cf = PowerSpectrumPlotter(101, 150, -1, 'AA').plot_power_spectra()