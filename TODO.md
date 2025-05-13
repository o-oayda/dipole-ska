# TODO

- Add dipole to all maps and recover power spectrum
    - **Geraint** will ask Sebastian
- Go through each map and see how (in)consistent they are with the input map
    - **Mali** to make pipeline (multiprocessing)
- Implement sky plot
    - **Oliver**
- Go through modulated_map function and comment
    - **Oliver**
- Summary statistics for outputs
    - **Mali** to do
    - BF comparing CMB dipole vs fitted one
    - Model 0: D_0, phi_0, theta_0 (fixed)
    - Model 1: Free dipole fit
    - `model1.evidence - model0.evidence`
- Can we test what happens if we set lmax=512 in both decomposition and reconstruction
    - Is the small scale stuff being truncated causing the negative cells in the map?
    - **Vasudev** to do