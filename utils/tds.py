import numpy as np
import scipy
from collections import namedtuple

_fresnel = namedtuple("FresnelCoefficientsForField","rp rs tp ts theta1 theta2 n1 n2")

def fresnel_coeff(n1=1,theta1=0,n2=2):
    """ returns the fresnel coefficients for (n1,theta1)->(n2,theta2); angle in rad """
    theta2 = np.arcsin(np.real_if_close(n1*np.sin(theta1) / n2))
    c1 = np.cos(theta1); s1 = np.sin(theta1)
    c2 = np.cos(theta2); s2 = np.sin(theta2)
    rs = (n1*c1-n2*c2)/(n1*c1+n2*c2)
    ts = 2*n1*c1/(n1*c1+n2*c2)
    rp = (n2*c1-n1*c2)/(n2*c1+n1*c2)
    tp = 2*n1*c1/(n2*c1+n1*c2)
    return _fresnel(rp=rp,rs=rs,tp=tp,ts=ts,theta1=theta1,theta2=theta2,n1=n1,n2=n2)


