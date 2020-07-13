import numpy as np 
import datastorage 
from openlab.oscilloscopes import lecroy
import matplotlib.pyplot as plt 


_non_linear_coeff = dict(
        GaP = 0.88e-12, # r41 in V/m
        ZnTe = 4.04e-12, # r41 in V/m
)

_n0_probe = dict(
        GaP = 3.20, # 800nm
        ZnTe = 2.85 # 800nm
)

_n0_THz = dict(
        ZnTe = 3.2,
        GaP = 3.3399,
        Ge = 4,
        Si = 3.42
        )


def field_from_EOS(dI_over_I=0.1,material="ZnTe",crystal_thickness=500e-6,
        probe_wavelength=800e-9):

        # fresnel coeff
        transmission_crystal = 2*1/(1+_n0_THz[material])

        dI_over_I = np.atleast_1d(dI_over_I)
        n0 = _n0_probe[material]
        nonlin = _non_linear_coeff[material]
        ETHz = np.arcsin(dI_over_I)*probe_wavelength/ (2*np.pi*n0**3*nonlin*transmission_crystal*crystal_thickness)
        ETHz = ETHz*1e-5 # from V/m to kV/cm
        if ETHz.shape[0]==1: ETHz = float(ETHz)
        return ETHz

def get_run(run):
    fname = run
    data = datastorage.read(fname)
    data = convert_to_V(data) # From ADU to V
    data.xdata *= 1e9
    return data  

def convert_to_V(data):
    y1data = data.y1data
    y2data = data.y2data
    y1info = lecroy.deserialize_descr(data.y1info[0])
    y2info = lecroy.deserialize_descr(data.y2info[0])
    if str(y1data.dtype) == "int16":
        y1data = y1data*y1info["vertical_gain"]-y1info["vertical_offset"]
        y2data = y2data*y2info["vertical_gain"]-y2info["vertical_offset"]
        data.y1data = y1data
        data.y2data = y2data
    return data

def mean_shots(data):
    mean = []
    for t in data:
        temp = 0
        for rep in t:
            temp += rep
        temp /= len(t)
        mean.append(temp)
    return np.asarray(mean)

def analyze_scan(run, plot = True, as_absolute_field = True, material = 'ZnTe', crystal_thickness = 500e-6):
    data = get_run(run)
    mean_y1 =[]
    mean_y2 = [] 
    for i in range(len(data.y1data)):
        mean_y1.append(mean_shots(data.y1data[i]))
        mean_y2.append(mean_shots(data.y2data[i]))
    mean_y1 = np.asarray(mean_y1)
    mean_y2 = np.asarray(mean_y2)
    eos = calculate_eos_signal(mean_y1, mean_y2, data.xdata)
    print(eos)
    eos_kV = field_from_EOS(eos,material = material, crystal_thickness = crystal_thickness)
    if as_absolute_field:
        if plot :
            plt.plot(-data.positions, eos_kV)
        return data.positions, eos_kV
    else:
        if plot:
            plt.plot(-data.positions, eos)
        return data.positions, eos

def finding_pulse(ydata, xdata):
    increase_threshold = 3*np.abs(np.min(ydata))
    central_peak = 0    
    for iy,y in enumerate(ydata):
        if iy == 0:
            continue
        if iy == len(ydata)-1:
            break
        if ydata[iy + 1] - y > increase_threshold and y - ydata[iy - 1] < increase_threshold:
            central_peak = xdata[iy]
    return central_peak

def integrate(ydata, xdata, start, end):

    dt = xdata[1] - xdata[0]
    dt *= 1e-9
    to_integrate = []
    integrale = 0
    for i,y in enumerate(ydata):
        if not xdata[i] < start or xdata[i] > end:
            to_integrate.append(y)
    for y in to_integrate:
        integrale += y*dt
    return integrale

def calculate_eos_signal(y1data, y2data, xdata):

    y1 = []
    y2 = []

    for time_1 in y1data:
        peak = finding_pulse(time_1,xdata)
        y1.append(integrate(time_1, xdata, peak -1 , peak + 12))

    for time_2 in y2data:
        peak = finding_pulse(time_2,xdata)
        y2.append(integrate(time_2, xdata, peak -1 , peak + 12))

    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    bkg = np.mean(y1[-10:-1]+y2[-10:-1])
    eos_signal = (y1-y2)/(y1+y2)

    return eos_signal