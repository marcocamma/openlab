import numpy as np
from matplotlib import pyplot as plt

import PyTango
import time
bpm = PyTango.DeviceProxy("//acs.esrf.fr:10000/srdiag/fast-bpm/id09")

WHAT_DEFAULT=["XAngleBuffer","ZAngleBuffer","XPosBuffer","ZPosBuffer"]

def get_bpm_server(name):
    v = bpm.read_attribute(name)
    return v

def get_bpm_server_value(name):
    return get_bpm_server.value

def has_same_timestamp(v1,v2):
    v1 = v1.time
    v2 = v2.time
    return v1.tv_nsec == v1.tv_nsec and v1.tv_sec==v2.tv_sec and v1.tv_usec == v2.tv_usec

def acquire(what=WHAT_DEFAULT,dt=2.5,polling=0.1,force_return_dict=True,concatenate=True):
    # if what is a list, it records several variables but there is no guarantee
    # that they will be aligned ...
    if isinstance(what,str):
        what = [what,]
    data_buffer = dict()
    for key in what:
        data_buffer[key] = [get_bpm_server(key),]
    t0 = time.time()
    while time.time()-t0<dt:
        t0_cycle = time.time()
        for key in what:
            temp = get_bpm_server(key)
            if not has_same_timestamp(temp,data_buffer[key][-1]):
                data_buffer[key].append(temp)
        time_before_polling = polling-(time.time()-t0_cycle)
        if time_before_polling>0: time.sleep(time_before_polling)
    data = dict()
    for key in what:
        if key.find("FFT")>0:
            data[key] = np.mean([d.value for d in data_buffer[key]],axis=0)
            data["_bpm_fft_xaxis"] = np.linspace(1,5074.5,len(data[key]))
        else:
            n = len(data_buffer[key])
            if concatenate:
                data["_bpm_xaxis"] = np.linspace(0,1*n,10150*n)
                data[key] = np.hstack([d.value for d in data_buffer[key]])
            else:
                data["_bpm_xaxis"] = np.linspace(0,1,10150)
                data[key] = np.asarray([d.value for d in data_buffer[key]])
    if not force_return_dict and len(what) == 1: data = data[what[0]]
    return data

def list():
    attributes = bpm.attribute_list_query()
    for a in attributes: print(a.name)

#plt.plot(get_bpm_server_value("XAngleFFT"))
#plt.plot(get_bpm_server_value("XAngleBuffer"))
