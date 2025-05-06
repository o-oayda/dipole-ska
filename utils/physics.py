import numpy as np

def omega_to_theta(omega):
    '''
    Convert solid angle in steradins to theta in radians for a cone section
    of a sphere.
    
    :param omega: Solid angle in steradians.
    :return: Angle in radians,
    '''
    return np.arccos(1 - omega / (2 * np.pi))