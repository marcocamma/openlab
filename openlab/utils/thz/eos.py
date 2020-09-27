import os
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.ndimage import measurements


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

def _fresnel_e_transmission(n1,n2):
    return 2*n1/(n1+n2)

def Efield_slab_transmission(material="Ge"):
    n = _n0_THz[material]
    t12 = _fresnel_e_transmission(1,n)
    t21 = _fresnel_e_transmission(n,1)
    return t12*t21


def find_integration_ranges(t,ytraces,bkg_time_range=(-4e-9,-2.5e-9),pulse_time_range=(-1.5e-9,5e-9),plot=False):
    """
    Parameters
    ----------
      ytraces: numpy array
        last index is time
        can be 1D (time), 2D (shotnum,time), or 3D (scanstep,shotnum,time)
    """

    dt = np.mean(np.diff(t))

    ytraces = np.asarray(ytraces)
    if ytraces.ndim > 3:
        raise ValueError("ytraces has to be 1D,2D or 3D")
    elif ytraces.ndim == 3:
        ytraces = ytraces.mean(axis=(0,1))
    elif ytraces.ndim == 2:
        ytraces = ytraces.mean(axis=0)



    # normalize
    baseline = np.percentile(ytraces,10) # baseline value
    peak = np.percentile(ytraces,99) # find 99% percentile as peak value
    ytraces = (ytraces-baseline)/(peak-baseline)

    # find pulses
    labels,npulses = measurements.label(ytraces>0.5)
    # take first index of each pulse
    idx = [ int(np.argwhere(labels==i)[0]) for i in range(1,npulses+1) ]

    # for each pulse define bkg and pulse integration ranges
    bkg_idx = []
    pulse_idx = []
    for i in idx:
        b = slice( int(i+bkg_time_range[0]/dt),int(i+bkg_time_range[1]/dt) )
        p = slice( int(i+pulse_time_range[0]/dt),int(i+pulse_time_range[1]/dt) )
        if b.start<0 or p.stop >len(t): continue
        bkg_idx.append( b )
        pulse_idx.append( p )

    if plot:
        plt.plot(t,ytraces)
        for b,p in zip(bkg_idx,pulse_idx):
            plt.axvspan(t[b.start],t[b.stop],color="#99d8c9",alpha=0.5)
            plt.axvspan(t[p.start],t[p.stop],color="#d95f02",alpha=0.5)

    return dict(
            bkgs_idx=bkg_idx,
            pulses_idx=pulse_idx,
            bkg_time_range=bkg_time_range,
            pulse_time_range=pulse_time_range
            )


def calc_pulse_integrals(t=None,ytraces=None,bkgs_idx=None,pulses_idx=None,**find_integration_ranges_kwg):
    if ytraces is None: raise ValueError("ytraces cannot be None")
    ytraces = np.asarray(ytraces)


    # makes ytraces most general (scanstep,shotnum,time)
    if ytraces.ndim == 1:
        ytraces = ytraces[np.newaxis,np.newaxis,:]
    elif ytraces.ndim == 2:
        ytraces = ytraces[np.newaxis,:,:]

    if bkgs_idx is None or pulses_idx is None:
        auto_idx = find_integration_ranges(t,ytraces,**find_integration_ranges_kwg)
        if bkgs_idx is None: bkgs_idx = auto_idx["bkgs_idx"]
        if pulses_idx is None: pulses_idx = auto_idx["pulses_idx"]


    integrals = np.zeros( (ytraces.shape[0],ytraces.shape[1],len(pulses_idx) ) )
    for ipulse,(bkg_idx,pulse_idx) in enumerate(zip(bkgs_idx,pulses_idx)):
        bkg = ytraces[:,:,bkg_idx].mean(axis=2)[:,:,np.newaxis]
        pulses = ytraces[:,:,pulse_idx]-bkg
        integrals[:,:,ipulse] = pulses.sum(axis=2)

    integrals = np.squeeze(integrals)

    return dict(integrals=integrals,bkg_idx=bkgs_idx,pulses_idx=pulses_idx)

def calc_eos_signal(y1integrals,y2integrals,pumped_pulse=3):
    """
    Parameters
    ----------
        y1integrals : dict or numpy array
            if dict the "integrals" key will be used
    """
    if isinstance(y1integrals,dict): y1integrals = y1integrals["integrals"]
    if isinstance(y2integrals,dict): y2integrals = y2integrals["integrals"]

    # just save some typing
    y1 = y1integrals
    y2 = y2integrals

    # make sure it is the most general case
    if y1.ndim == 2: y1 = y1[np.newaxis,:,:]
    if y2.ndim == 2: y2 = y2[np.newaxis,:,:]

    ref1 = np.concatenate( (y1[:,:,:pumped_pulse],y1[:,:,pumped_pulse+1:]),axis=2 )
    ref2 = np.concatenate( (y2[:,:,:pumped_pulse],y2[:,:,pumped_pulse+1:]),axis=2 )

    y1 = y1[:,:,pumped_pulse] / ref1.mean(axis=2)
    y2 = y2[:,:,pumped_pulse] / ref2.mean(axis=2)


    dy = y2-y1
    s  = y2+y1

    data = dy/s

    return np.squeeze(np.median(data,axis=1)),dy,s,y1,y2


def analyze_scan(fname,as_absolute_field=True,plot=True,**field_from_EOS_kwg):
    data = h5py.File(fname,"r+")
    delay = -data["positions"].value
    y1data = data["y1data"].value
    y2data = data["y2data"].value
    t = data["xdata"].value
    integral1 = calc_pulse_integrals(t=t,ytraces=y1data)
    integral2 = calc_pulse_integrals(t=t,ytraces=y2data)
    eos,diff_y,sum_y,y1,y2 = calc_eos_signal(integral1,integral2)

    mean_sum_y=np.mean(sum_y,axis=1)
    mean_y1=np.mean(y1,axis=1)
    mean_y2=np.mean(y2,axis=1)
    if as_absolute_field:
        eos_absolute = field_from_EOS(eos,**field_from_EOS_kwg)
    else:
        eos_absolute = None

    if plot:
        info = "scan %s"%(fname)
        if eos_absolute is None:
            plt.figure()
            plt.plot(delay,eos,label=info)
            plt.ylabel("EOS signal a.u.")
            plt.xlabel('t in ps')

            plt.figure()
            plt.plot(delay,mean_sum_y,label='sum y1+y2')
            plt.ylabel("sum photodiodes a.u.")
            plt.xlabel('t in ps')
            
            fig,ax=plt.subplots(2,1)
            ax[0].plot(delay,mean_y1,label='y1')
            
            ax[1].plot(delay,mean_y2,label='y2')
            ax[1].set_xlabel('t ps')

            
            
        else:
            plt.plot(delay,eos_absolute,label=info)
            plt.ylabel("E (kV/cm)")
            plt.xlabel('t in ps')
        plt.title(info)
    #data["eos"] = eos
    #if eos_absolute is not None: data["eos_absolute"] = eos_absolute
    return delay,eos_absolute,eos,sum_y,y1,y2#,data

def analyze_scan_2(fname,as_absolute_field=True,plot=True, plot_fft = False,**field_from_EOS_kwg):
    data = h5py.File(fname,"r+")
    delay = -data["positions"].value
    y1data = data["y1data"].value
    y2data = data["y2data"].value
    t = data["xdata"].value
    integral1 = calc_pulse_integrals(t=t,ytraces=y1data)
    integral2 = calc_pulse_integrals(t=t,ytraces=y2data)
    eos = calc_eos_signal(integral1,integral2)
    if as_absolute_field:
        eos_absolute = field_from_EOS(eos,**field_from_EOS_kwg)
    else:
        eos_absolute = None

    if plot:
        info = "scan %s"%(fname)
        if eos_absolute is None:
            plt.plot(delay,eos,label=info)
            plt.ylabel("EOS signal a.u.")
            plt.xlabel('t in ps')
        else:
            plt.plot(delay,eos_absolute,label=info)
            plt.ylabel("E (kV/cm)")
            plt.xlabel('t in ps')
        plt.title(info)

    if plot_fft:
        info = "scan %s"%(fname)
        figure, ax = plt.subplots(2,1)
        ax[0].plot(delay,eos_absolute, label = info)
        E_f, freq = fft(eos_absolute, delay)
        ax[1].plot(freq, np.abs(E_f))

    
    #data["eos"] = eos
    #if eos_absolute is not None: data["eos_absolute"] = eos_absolute
    return delay,eos#,data



def fft(E,t): 
    E_cut = E[t<25]
    E_freq_full = np.fft.fft(E_cut)
    freq_full = np.fft.fftfreq(len(E_cut),t[1]-t[0])
    freq = freq_full[freq_full>0]
    E_freq = E_freq_full[freq_full>0]
    return E_freq, freq

def get_run(run):
    fname = 'scan%04d.h5'%run
    return fname

def plot_focus(first_run,last_run, step, material = 'ZnTe', crystal_thickness = 500e-6):
    runs = np.arange(first_run, last_run, step = step)
    total = len(runs)
    figure,ax = plt.subplots(total,1)
    for number_run, run in enumerate(runs):
        fname = get_run(run)
        data = analyze_scan(fname, plot = False, material = material, crystal_thickness= crystal_thickness)
        ax[number_run].plot(data[0],data[1])