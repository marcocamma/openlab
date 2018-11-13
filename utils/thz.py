import os
import time
import numpy as np

_non_linear_coeff = dict(
        GaP = 0.88e-12, # r41 in V/m
        ZnTe = 4.04e-12, # r41 in V/m
)

_n0 = dict(
        GaP = 3.20, # 800nm
        ZnTe = 2.85 # 800nm
)



def field_from_EOS(dI_over_I=0.1,material="ZnTe",crystal_thickness=500e-6,
        transmission_crystal=0.5,probe_wavelength=800e-9):

        dI_over_I = np.atleast_1d(dI_over_I)
        n0 = _n0[material]
        nonlin = _non_linear_coeff[material]
        ETHz = np.arcsin(dI_over_I)*probe_wavelength/ (2*np.pi*n0**3*nonlin*transmission_crystal*crystal_thickness)
        ETHz = ETHz*1e-5 # from V/m to kV/cm
        if ETHz.shape[0]==1: ETHz = float(ETHz)
        return ETHz

