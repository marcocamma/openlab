import numpy as np
import time
import pathlib
from openlab.labs.mllab import thzsetup
from openlab.generic import scan
from datastorage import DataStorage as  ds

xyz = thzsetup.xyz
scope = thzsetup.scope

folder = pathlib.Path("/data/thz/2019.02")

def acquire():
    scope.acquire_for_time(5)
    pars = scope.get_all_pars_for_measurement(1)
    av = pars["AVG"].v
    sigma = pars["SIGMA"].v
    trace = scope.get_waveform(1)[1]
    t = scope.get_xaxis()
    tosave = dict(av=av,sigma=sigma,trace=trace,_t=t)
    return tosave

def acquire_test():
    trace = scope.get_waveform(1)[1]
    t = scope.get_xaxis()
    tosave = dict(trace=trace,_t=t)
    return tosave


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
