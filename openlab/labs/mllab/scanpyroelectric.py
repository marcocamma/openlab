import numpy as np
import time
import pathlib
from openlab.labs.mllab import thzsetup
from openlab.generic import scan
from datastorage import DataStorage as  ds
import os 


xyz = thzsetup.xyz
scope = thzsetup.scope

folder = pathlib.Path("/data/thz/2020.02/pyroelectric")

def acquire():
    scope.acquire_for_time(5)
    pars = scope.get_all_pars_for_measurement(1)
    av = pars["AVG"].v
    sigma = pars["SIGMA"].v
    trace = scope.get_waveform(4)[1]
    t = scope.get_xaxis()
    tosave = dict(av=av,sigma=sigma,trace=trace,_t=t)
    return tosave

def acquire_test():
    trace = scope.get_waveform(1)[1]
    t = scope.get_xaxis()
    tosave = dict(trace=trace,_t=t)
    return tosave

def save_trace(nshots):
    if not os.path(folder / 'run_number.npz'):
        run_number = 0
        np.savez('run_number', run_number = run_number)
    run_number = np.load('run_number.npz')
    run_number = run_number['run_number']
    fname = "scan%04d.h5"%scannum
    fname = folder / fname
    av = []
    trace = []
    sigma = []
    i = 0
    while i < nshots:
        temp = acquire()
        av.append(temp['av'])
        t = temp['_t']
        trace.append(temp['trace'])
        sigma.append(temp['sigma'])
        i += 1
    to_save = dict(av = av, trace = trace, sigma = sigma, t = t)
    ds.save(fname, to_save)



def scan1d(scannum,motor,start,stop,N=20):
    fname = "scan%04d.h5"%scannum
    fname = folder / fname
    tosave = scan.ascan(motor,start,stop,N,acquire=acquire,fname=fname)
    return tosave

def scan2d(scannum,s1,e1,n1,s2,e2,n2,motor1=xyz.y,
        motor2 = xyz.x, t=5):
    fname = "scan%04d.h5"%scannum
    fname = folder / fname
    tosave = scan.a2scan(motor1,s1,e1,n1,motor2,s2,e2,n2,
            acquire=acquire,fname=fname)
    return tosave

def scan2d_snake(scannum,s1,e1,n1,s2,e2,n2,motor1=xyz.y,
        motor2 = xyz.x, t=5):
    fname = "scan%04d.h5"%scannum
    fname = folder / fname
    tosave = scan.a2scan_snake(motor1,s1,e1,n1,motor2,s2,e2,n2,
            acquire=acquire,fname=fname)
    return tosave

def two_scan1d(run, start,stop, N = 20):
    thzsetup.xyz.x.move(0)
    thzsetup.xyz.y.move(0)
    thzsetup.xyz.y.wait()
    scan1d(run, thzsetup.xyz.x,start,  stop, N)
    thzsetup.xyz.x.move(0)
    thzsetup.xyz.x.wait()
    scan1d(run + 1 , thzsetup.xyz.y,start,  stop, N)