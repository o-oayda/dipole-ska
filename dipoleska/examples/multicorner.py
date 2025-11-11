from dipoleska.utils.posterior import Posterior
import matplotlib.pyplot as plt


post = Posterior(114)
post2 = Posterior(115)
post.add_comparison_run(run=post2)

post.corner_plot(backend='getdist', coordinates=['equatorial', 'galactic'])
post.sky_direction_posterior(coordinates=['equatorial', 'galactic'])
plt.show()
