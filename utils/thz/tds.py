import numpy as np
import scipy
from collections import namedtuple
from . import eos
import matplotlib.pyplot as plt 
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





#from here, script changed by Guenole, no change has been made above.
def fft(time, field, time_cut,cut=True,):
    # Return the positive frequency and the corresponding field
    dt =time[1] - time[0]
    if cut:
        field = field[ time < time_cut]
    else:
        field=field
    field_transform = np.fft.fft(field)
    freq = np.fft.fftfreq(len(field), dt)
    freq_return  = freq[freq > 0] 
    field_transform_return = field_transform[freq > 0]
    return freq_return, field_transform_return

def plot_FT(run,timecut=7,cut=True):
    time, field, trash = eos.analyze_scan(run, as_absolute_field=True, material = 'ZnTe', crystal_thickness = 500e-6, plot = False)
    freq, tranform_field = fft(time, field, timecut,cut)
    plt.plot(freq, np.abs(tranform_field))
    plt.xlim(0,4)

def FTTDS_Lucas(run_on,run_off):
    #need same time arrays
    time,eos_abs_on,eos_on=eos.analyze_scan(run_on, as_absolute_field=True, material = 'ZnTe', crystal_thickness = 500e-6)
    time_off,eos_abs_off,eos_off=eos.analyze_scan(run_off,as_absolute_field=True, material = 'ZnTe', crystal_thickness = 500e-6)
    time_cut=time[time<7]
    n=time_cut.size
    dt=time[0]-time[1]
    Freq=np.fft.rfftfreq(n,d=dt)
    FTEon=np.fft.rfft(eos_on[time<7],n)
    FTEoff=np.fft.rfft(eos_off[time<7],n)
    off_cut=FTEoff[Freq>0]
    on_cut=FTEon[Freq>0]
    print(time)
    print(time_cut)
    Freq_cut=Freq[Freq>0]
    on_norm=on_cut/np.amax(on_cut)
    off_norm=off_cut/np.amax(off_cut)
    H=on_norm/off_norm
    return on_norm,off_norm,H,Freq,Freq_cut

def FTTDS_old(run,timecut=7,norm=True):
#need same time arrays
    time,eos_abs,eos_=eos.analyze_scan(run, as_absolute_field=True, plot=False,material = 'ZnTe', crystal_thickness = 500e-6)
    time_cut=time[time<timecut]
    n=time_cut.size
    dt=time[0]-time[1]
    Freq=np.fft.fftfreq(n,d=dt)
    FTE=np.fft.fft(eos_[time<timecut],n)
    FT_cut=FTE[Freq>0]
    print(time)
    print(time_cut)
    Freq_cut=Freq[Freq>0]

    FT_abs=np.abs(FT_cut)
    FT_norm=FT_cut/np.amax(FT_abs)
    #H=on_norm/off_norm
    if norm:
        return FT_norm,FT_cut,Freq_cut,eos_abs,time
    else:
        return FT_abs,FT_cut,Freq_cut,eos_abs,time


 # FIXME: a changer --> le programme filling FT marche mais attention ! il faut revenir dans espace frequentiel

def plot_TDS(run,nbpoint=10,timecut=7,norm=True):
    ft_nooff_norm,ft_nooff,fr,eos_nooff,time=FTTDSbis(run,nbpoint,timecut)
    #plt.figure()
    if norm:
        plt.plot(fr,np.abs(ft_nooff_norm),label=run)
    else:
        plt.plot(fr,ft_nooff,label=run)
    plt.xlim(0,4)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    #plt.figure()
    #plt.plot(time,eos_abs)



def FTTDSbis(run,nbpoint=10,timecut=7,norm=True):
#need same time arrays
    time,eos_abs,eos_=eos.analyze_scan(run,as_absolute_field=True, plot=False,material = 'ZnTe', crystal_thickness = 500e-6)
    time_cut=time[time<timecut]
    n=time_cut.size
    dt=time[0]-time[1]
    Freq=np.fft.fftfreq(n,d=dt)
    eos_nooff=[]
    for i in eos_abs:
        eos_nooff.append(i-np.mean(eos_abs[:nbpoint]))
    eos_nooff=np.asarray(eos_nooff)
    FTE=np.fft.fft(eos_abs[time<timecut],n)
    FTEnooff=np.fft.fft(eos_nooff[time<timecut],n)
    FT_cut=FTE[Freq>0]
    FTnooff_cut=FTEnooff[Freq>0]
    print(time)
    print(time_cut)
    Freq_cut=Freq[Freq>0]

    FT_abs=np.abs(FT_cut)
    FT_norm=FT_cut/np.amax(FT_abs)
    FTnooff_abs=np.abs(FTnooff_cut)
    FTnooff_norm=FTnooff_cut/np.amax(FTnooff_abs)
    #H=on_norm/off_norm
    if norm:
        return FTnooff_norm,FTnooff_abs,Freq_cut,eos_nooff,time
    else:
        return FT_abs,FT_cut,Freq_cut,eos_abs,time




def H_treatment(run_with,timecut_with,run_without,timecut_without):
    FT_norm_with,FT_with,Freq_cut,eos_abs,time=FTTDS(run_with,timecut_with)
    FT_norm_without,FT_without,Freq_cut,eos_abs,time=FTTDS(run_without,timecut_without)
    t=time

    print(t)
    Without,With,GoodWithout,time1,time2=filling_FT(FT_norm_without,FT_norm_with,t,timecut_without,timecut_with)

    return FT_norm_with,FT_norm_without,t
    #H=FT_norm_with/FT_norm_without
    #ph=np.unwrap(np.angle(H))
    #lt.figure()
    #plt.plot(Freq_cut,np.abs(FT_norm_with))
    #plt.plot(Freq_cut,np.abs(FT_norm_without))
    #plt.figure()
    #plt.yscale('log')
    #plt.plot(Freq_cut,np.abs(H))
    #plt.xlim(0.1,2)
    #plt.figure()
    #plt.plot(Freq_cut,ph)
    #plt.xlim(0.1,2)
    #return Freq_cut ,ph

def filling_FT(array1,array2,time,timecut1,timecut2):
    time1=time[time<timecut1]
    time2=time[time<timecut2]
    arcut1=array1[time<timecut1]#need t be the shortest
    arcut2=array2[time<timecut2]
    list1=[]
    print(time1)
    print(time2)
    
    if len(arcut1)<len(arcut2):
        for i in arcut1:
            list1.append(i)
        for i in range(0,len(time2)-len(time1)):
            list1.append(1)
    list1=np.asarray(list1)
    return arcut1,arcut2,list1,time1,time2

def test_filling():
    T=1.0
    time=np.arange(6,12,0.01)
    array1=np.sin(2*np.pi/T*time)
    array2=np.cos(2*np.pi/T*time)
    tc1=6.9
    tc2=7.9

    ar1,ar2,li1,t1,t2=filling_FT(array1,array2,time,tc1,tc2)
    plt.figure()
    plt.plot(t2,ar2)
    plt.plot(t2,li1)
    return 
