import numpy as np 
import datastorage 
from openlab.oscilloscopes import lecroy
import matplotlib.pyplot as plt 

def get_run(run):
    fname = run
    data = datastorage.read(fname)
    data = convert_to_V(data) # From ADU to V
    data.xdata *= 1e9
    return data  

def convert_to_V(data):
    y1data = data.ydata
    y1info = lecroy.deserialize_descr(data.yinfo[0])
    if str(y1data.dtype) == "int16":
        y1data = y1data*y1info["vertical_gain"]-y1info["vertical_offset"]
        data.ydata = y1data
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

def analyze_scan(run, plot = True):
    data = get_run(run)
    mean_y1 =[]
    for i in range(len(data.ydata)):
        mean_y1.append(mean_shots(data.ydata[i]))
    mean_y1 = np.asarray(mean_y1)
    eos = calculate_signal(mean_y1, data.xdata)
    if plot:
        plt.plot(data.positions, eos)
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
    for y in ydata:
        integrale += y*dt
    return integrale

def calculate_signal(y1data, xdata):

    y1 = []

    for time_1 in y1data:
        peak = finding_pulse(time_1,xdata)
        y1.append(integrate(time_1, xdata, peak -1 , peak + 12))

    y1 = np.asarray(y1)

    return y1