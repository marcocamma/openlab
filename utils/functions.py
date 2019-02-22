import numpy as np
from scipy.special import erf
SQRT2 = np.sqrt(2)

def slitgauss(x,a=1,x0=0,sig=1,width=1,normalize=False):
    """ Transmission of a gaussian with sigma of `sig`
    trough a slit of a given width """
    sig = sig*SQRT2 # to take care the way erf is normalized
    arg1 = (x-(x0-width/2))/sig
    arg2 = (x-(x0+width/2))/sig
    y = a/2*( erf(arg1)-erf(arg2) )
    if normalize: y /= y.max()
    return y
