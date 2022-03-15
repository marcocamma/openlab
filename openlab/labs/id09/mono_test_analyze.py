import datastorage
import pathlib
import numpy as np
#from openlab.utils import mcutils
from matplotlib import pyplot as plt
from scipy.signal import welch
from scipy.signal.windows import hann
from scipy import signal


import lmfit

CH=12.3984192976
_d_111 = 3.13465698029
_d_333 = _d_111/3

def digital_filter(x,y,f=1000,order=10,kind="lowpass"):
    dt = x[1]-x[0]
    fs = 1/dt
    sos = signal.butter(order, f, kind, fs=fs, output='sos')
    return signal.sosfiltfilt(sos,y)

def lowpass(x,y,f=1000,order=10):
    return digital_filter(x,y,f=f,order=order,kind="lowpass")

def highpass(x,y,f=1000,order=10,add_average=True):
    temp = digital_filter(x,y,f=f,order=order,kind="highpass")
    const = y.mean() if add_average else 0
    return temp+const

def bandpass(x,y,freq=[99,101],order=10,add_average=True):
    """ freq can be a tuple (fmin,fmax) or a list of tuples if multiple bands """
    freqs = np.atleast_2d(freq)
    temp = np.zeros_like(x)
    for band in freqs:
        temp += digital_filter(x,y,f=band,order=order,kind="bandpass")
    const = y.mean() if add_average else 0
    return temp+const

def bandstop(x,y,freq=[99,101],order=10):
    """ freq can be a tuple (fmin,fmax) or a list of tuples if multiple bands """
    freqs = np.atleast_2d(freq)
    temp = np.zeros_like(x)
    for band in freqs:
        temp += digital_filter(x,y,f=band,order=order,kind="bandstop")
    return temp

def average_every(x,n=10):
    nout = x.shape[0] // n
    N = nout*n
    return x[:N].reshape(nout,n).mean(1)

def integrated_rms_noise(x,y,normalize=False,freqs=np.logspace(0,4,100)):
    if normalize: y = y/y.mean()
    noise = [lowpass(x,y,f).std() for f in freqs]
    noise = np.asarray(noise)
    return freqs,noise

def amplitude_spectral_density(x,y,normalize=True):
    if normalize: y = y/y.mean()
    dt = x[1]-x[0]
    fs = 1/dt
    f,asd=welch(y,fs=fs,nperseg=len(y))
    return f,asd


def si333_to_rad(energy):
    a = CH/(2*_d_333*energy)
    a = np.arcsin(a)
    return a

def si333_to_deg(energy):
    a = si333_to_rad(energy)
    a = np.rad2deg(a)
    return a


def read(fname,fmax=None,save=True,normalize_srcur=True):
    fname2 = pathlib.Path(fname).with_suffix(".more.h5")
    if fname2.is_file():
        data = datastorage.read(fname2)
    else:
        data = datastorage.read(fname)
    if fmax is not None:
        if not "fmax" in data or data["fmax"] != fmax:
            dx = np.mean(np.diff(data.xdata))
            data.y1_ff = lowpass(data.xdata,data.y1,fmax)
            data.y2_ff = lowpass(data.xdata,data.y2,fmax)
            data.fmax=fmax
            if save: data.save(fname2)
    if "srcur" in data and normalize_srcur:
        for i in range(len(data.y1)):
            data.y1[i] /= data.srcur[i]
            data.y2[i] /= data.srcur[i]
            data.y1_ff[i] /= data.srcur[i]
            data.y2_ff[i] /= data.srcur[i]
    return data



def test_filter_noise():
    noise = read("data/run007.h5",fmax=None)
    dx = np.mean(np.diff(noise.xdata))
    def testf1(fmax=1e3):
        f=mcutils.FFTlowpass(noise.y1[0],dx=dx,fmax=fmax)
        return f.apply(noise.y1[0]).std()

    def testf2(fmax=1e3):
        f=mcutils.FFTlowpass(noise.y2[0],dx=dx,fmax=fmax)
        return f.apply(noise.y1[0]).std()

    f=np.linspace(50,10000,101)
    plt.plot(f,[testf1(fi) for fi in f],label="ch1")
    plt.plot(f,[testf2(fi) for fi in f],label="ch2")
    plt.legend()
    plt.xlabel("cut off freq")
    plt.ylabel("RMS noise (no beam, V)")
    plt.title("openlab.run007")
    plt.grid()

def remove_linear_trend_curve(x,y):
    poly = np.polyfit(x,y,1)
    y = y-poly[0]*x
    return y


def remove_linear_trend(data):
    y1 = data.y1_ff
    y2 = data.y2_ff
    t = data.xdata
    e = data.positions
    # remove 'drifts' due to mono moving
    n = y1.shape[0]
    for i in range(n):
        data.y1_ff_lin[i] = remove_linear_trend_curve(t,y1[i])
        data.y2_ff_lin[i] = remove_linear_trend_curve(t,y2[i])
    return data



def analyze_stability_run(data,fmax=2e3,use_monoangle=False,use_ratio=True,factor_std=100):
    if isinstance(data,str):
        data = read(data,fmax=fmax,save=True)

    if "y1_ff" in data and fmax is not None:
        print("in analyze_stability_run, will use frequency filtered data")
        y1 = data.y1_ff
        y2 = data.y2_ff
    else:
        y1 = data.y1
        y2 = data.y2
    t = data.xdata
    if use_monoangle:
        x = data.mono
    else:
        x = data.positions
    # remove 'drifts' due to mono moving
    n = y1.shape[0]
    for i in range(n):
        y1[i] = remove_linear_trend_curve(t,y1[i])
        y2[i] = remove_linear_trend_curve(t,y2[i])
    if use_ratio:
        ratio = y2/y1
    else:
        ratio = y1
    std = ratio.std(axis=1)
    mean = ratio.mean(axis=1)
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].set_title(data.filename)
    if use_ratio:
        ax[0].set_ylabel("abs(derivative of transmission)")
        label = "average pd3/pd1"
    else:
        ax[0].set_ylabel("abs(derivative of signal)")
        label = "pd1"
    ax[1].plot(x,mean,label=label)
    ax[1].plot(x,std*factor_std,label=f"{factor_std}×std")
    ax[0].plot(x,np.abs(np.gradient(mean,x)))
    ax[1].legend()
    for a in ax: a.grid()
    if use_monoangle:
        ax[1].set_xlabel("mono(deg)")
    else:
        ax[1].set_xlabel(str(data.info.motors[0]))

def find_slope_at_edge(data):
    x = data.mono
    r = np.mean(data.y2/data.y1,axis=1)
#    idx = (x>17.271) & (x<17.274)
    idx = (x>17.271) & (x<17.274)
    poly = np.polyfit(x[idx],r[idx],1)
    slope = poly[0]
    return slope

def freq_analysis(data,scanpoints=1,fmax=1e4,use_monoangle=True,asd_scale=7,xlim=200,time_y_scale=0.1):
    x = data.xdata
    if use_monoangle:
        motor = data.mono
        motor_name = "bragg_angle"
    else:
        motor = data.positions
        motor_name = data.info.motors[0]
    if isinstance(scanpoints,int): scanpoints = (scanpoints,)

    fig,axes=plt.subplots(4,len(scanpoints),figsize=[5*len(scanpoints),8],squeeze=False,sharey="row")#,sharex="col",sharey=True)
    

    for i,scanpoint in enumerate(scanpoints):
        ax=axes[:,i]
        y1= remove_linear_trend_curve(x,data.y1[scanpoint])
        y2= remove_linear_trend_curve(x,data.y2[scanpoint])
        dx = np.mean(np.diff(x))
        y1_ff = lowpass(x,y1,fmax)
        y2_ff = lowpass(x,y2,fmax)
        r = y2/y1
        r_ff = y2_ff/y1_ff
        dx = np.mean(np.diff(x))
        title = str(data.filename)+" scanpoint %d"%scanpoint
        title += f"\n({motor_name} = {motor[scanpoint]:.4f})"
        title += f" (fmax={int(fmax)})"
        ax[0].set_title(title)
        ax[0].plot(x,r_ff/r_ff.mean())
        ax[0].set_xlim(0,0.2)
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylim(1-time_y_scale/2,1+time_y_scale/2)
        titles = "pd3/pd1 pd3 pd1".split()
        for j,d in enumerate((r,y2,y1)):
            f,fft=amplitude_spectral_density(x,d,normalize=True)
            ax[j+1].plot(f,fft*np.power(10,asd_scale))
            axes[j+1,0].set_ylabel(titles[j]+r" ASD (×10$^{"+str(asd_scale)+r"}$)")
            ax[j+1].set_xlim(0,xlim)
            ax[j+1].set_ylim(0,200)
        for j in (1,2):
            plt.setp( ax[j].get_xticklabels(), visible=False)
        ax[-1].set_xlabel("freq (Hz)")
    axes[0,0].set_ylabel("pd3/pd1 (normalized)")
    for a in axes.ravel():
        a.grid()
    plt.tight_layout()


def fig_stability1():
    data=read("2022_02_ccm_test/run018.h5",fmax=2e3)
    idx = [33,43,50]
    freq_analysis(data,scanpoints=idx,fmax=2e3)
    plt.savefig("run018_freq.pdf",transparent=True)
    analyze_stability_run(data,factor_std=100,use_monoangle=True)
    fig = plt.gcf()
    ax = fig.axes[1]
    for i in idx:
        ax.axvline(data.mono[i],color="0.5")
    plt.savefig("run018_stability.pdf",transparent=True)

    d = data

    fig,ax=plt.subplots(1,1,figsize=[7,4],sharex=True)
    x = d.xdata
    idx = (x>0.12) & (x<0.15)
    x = x[idx]
    y1 = d.y1_ff[-1,idx]
    y2 = d.y2_ff[-1,idx]
    ax.plot(x,y1/y1.mean(),label=f"pd1/{y1.mean():.2f}")
    ax.plot(x,y2/y2.mean(),label=f"pd3/{y2.mean():.2f}")
    ax.set_ylabel("normalized amplifier output")
    ax.set_xlabel("time (s)")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig("run018_traces.pdf",transparent=True)

def fig_stability2():
    fig,ax=plt.subplots(1,1)
    data=read("2022_02_ccm_test/run012.h5",fmax=1e3)
    i = -1
    plt.plot(d18.xdata,d18.y1_ff[i],d18.xdata,d18.y2_ff[i]*2.395)
    plt.legend(["pd1","2.395×pd3"])
    ax.set_ylabel("pd1 (V)")
    ax.grid()
    ax[-1].set_xlabel("time (s)")
    plt.xlim(0,0.3)
    plt.tight_layout()
    plt.savefig("run012_traces.pdf",transparent=True)


def fig_stability3():
    data=read("2022_02_ccm_test/run018.h5")
    x = data.xdata
    y = data.y1[3]
    # normalize
    y = y/y.mean()
    y = y-1

    y = lowpass(x,y,f=2000)

    fig,ax=plt.subplots(2,3,figsize=[14,6])

    ax[0,0].plot(x,y)
    ax[0,0].set_ylabel("pd1(norm)-1")
    ax[0,0].set_title(f"run018, {len(x)} points, 2 KHz low pass")

    ax[1,0].plot(x,y)
    for band in ( (1,90), (90,110), (300,450) ):
        ax[1,0].plot(x,bandpass(x,y,band),label=f"{band[0]}-{band[1]}")
    ax[1,0].set_xlim(0,0.1)
    ax[1,0].legend(title='bandpass (Hz)')
    ax[1,0].set_xlabel("time (s)")

    f,asd = amplitude_spectral_density(x,y,normalize=False)
    ax[0,1].plot(f,asd*100)
    ax[0,1].set_xlim(0,1500)
    ax[0,1].set_ylabel("amplitude spectral\ndensity (%/$\sqrt{Hz}$)")

    idx = f<=90

    ax[0,2].plot(f[idx],asd[idx]*100)
    ax[0,2].set_xlim(0,90)
    ax[0,2].set_ylabel("amplitude spectral\ndensity (%/$\sqrt{Hz}$)")

    f,noise = integrated_rms_noise(x,y,normalize=False)
    ax[1,1].plot(f,noise*100)
    ax[1,1].set_ylabel("integrated RMS noise (×100)")
    ax[1,1].set_xlabel("f (Hz)")
    ax[1,1].set_xlim(0,1500)
    ax[1,1].set_ylim(0,None)

    idx = f<=90
    ax[1,2].plot(f[idx],noise[idx]*100)
    ax[1,2].set_ylabel("integrated RMS noise (×100)")
    ax[1,2].set_xlabel("f (Hz)")
    ax[1,2].set_xlim(0,90)
    ax[1,2].set_ylim(0,None)

    for a in ax.ravel(): a.grid()
    plt.tight_layout()
    plt.savefig("run018_traces2.pdf",transparent=True)





def settling():
    calib = read("2022_02_ccm_test/run005.h5")
    E = calib.positions
    a = si333_to_deg(E)
    y1 = calib.y1.mean(1)
    idx = (E>19.975) & (E<20.00)
    p = np.polyfit(a[idx],y1[idx],1)

    fig,ax=plt.subplots(2,1)

    ax[0].plot(a,y1,"o")
    ax[0].plot(a[idx],np.polyval(p,a[idx]),label=f"slope = {p[0]:.1f}V/deg")
    ax[0].legend()
    ax[0].set_ylabel("pd1 (V)")
    ax[0].set_xlabel("mono angle (deg)")


    d = read("2022_02_ccm_test/run004.h5")

    # average every "av"
    av = 1000
    y = d.y1.reshape( (d.y1.shape[0]//av,av) ).mean(1)
    x = d.xdata.reshape( (d.y1.shape[0]//av,av) ).mean(1)

    x = x+8.9
    idx = (x>-1) & (x<0)
    yoff = y[idx].mean()
    y = (y-yoff)/p[0]
    ax[1].plot(x,y)

    def myf(x,a1,a2,t1,t2,t0):
        ret = np.zeros_like(x)
        idx = x>t0
        ret[idx] = a1*(1-np.exp(-(x[idx]-t0)/t1))+\
                   a2*(1-np.exp(-(x[idx]-t0)/t2))
        return ret

    m = lmfit.model.Model(myf)
    
    pars = m.make_params(
            a1=8e-3,t1=0.3,
            a2=2e-3,t2=4,
            t0=0.1
            )

    fit = m.fit(y,x=x,params=pars)
    v = fit.best_values
    f1 = v['a1']/(v["a1"]+v["a2"])*100
    f2 = 100-f1
    txt = f"exp1 = {f1:.0f}% (τ={v['t1']:.2f}s)\n\nexp2 = {f2:.0f}% (τ={v['t2']:.2f}s)"
    ax[1].text(8,5e-3,txt,fontsize=10)

    ax[1].plot(x,fit.best_fit)

    def V_to_deg(v):
        return (v-yoff)/p[0]

    def deg_to_V(deg):
        return deg*p[0]+yoff

    aaa= ax[1].secondary_yaxis('right', functions=(deg_to_V,V_to_deg))
    ax[1].set_ylabel("Δangle (deg)")
    aaa.set_ylabel("pd1 (V)")
    ax[1].set_xlabel("time (s)")
    for a in ax: a.grid()
    plt.tight_layout()
    plt.savefig("run004.pdf",transparent=True)

def expand(v,l):
    v = np.concatenate( [np.ones(len(l[i]))*v[i] for i in range(len(l))])
    return v

def MIM(average_every=10000):
    average_every = int(average_every)
    d = read("2022_02_ccm_test/run019.h5",fmax=None)
    fig,ax=plt.subplots(3,1,figsize=[6,8])
    idx = (41,42,43)
    #idx = (73,74,75)
    #idx = np.arange(len(d.mono)) 
    r = d.y2.mean(1)/d.y1.mean(1)
    x = d.mono
    deriv = np.gradient(r,x)
    ax[0].plot(x,deriv,"o",color="0.5")
    ax[1].plot(x,r,"o",color="0.5")
    for i in idx:
        ax[1].plot(x[i],r[i],"o")
        temp = d.y2[i]/d.y1[i]
        y = temp.reshape( (temp.shape[0]//average_every, average_every) ).mean(1)
        #_x = d.xdata.reshape( (d.xdata.shape[0]//average_every, average_every) ).mean(1)
        _x = np.ones_like(y)*(d.mono[i]-d.mono[idx[0]])
        _x = np.deg2rad(_x)*1e6
        #d.xdata.reshape( (d.xdata.shape[0]//average_every, average_every) ).mean(1)
        print(_x)
        #_x += (d.xdata[-1]-d.xdata[0])*i
        #_x = np.arange(nlast,nlast+len(y))
        ax[2].plot(_x,y,".")
    ax[0].set_xlabel("mono angle")
    ax[0].set_ylabel("dtransmission/dmono")
    ax[1].set_xlabel("mono angle")
    ax[1].set_ylabel("transmission")
    ax[0].set_xlim(x.mean()-1e-3,x.mean()+1e-3)
    ax[1].set_xlim(x.mean()-1e-3,x.mean()+1e-3)
    ax[2].set_xlabel("Δmono angle (μrad)")
    ax[2].set_ylabel("transmission")
    for a in ax: a.grid()
    plt.tight_layout()
    plt.savefig("run019_mim.pdf",transparent=True)


def fig_pointing_stability():

    fmax = 200

    data=read("2022_02_ccm_test/run023.h5",fmax=fmax)
    idx = [15,20,25]
    idx = [5,8,15]
    freq_analysis(data,scanpoints=idx,fmax=fmax,use_monoangle=False,asd_scale=4,xlim=50,time_y_scale=0.3)
    plt.savefig("run023_freq_analysis.pdf",transparent=True)
    analyze_stability_run(data,fmax=fmax,factor_std=10,use_monoangle=False)
    fig = plt.gcf()
    ax = fig.axes[1]
    for i in idx:
        ax.axvline(data.positions[i],color="0.5")
    plt.savefig("run023_stability.pdf",transparent=True)

    x = data.xdata
    y1 = data.y1_ff[8]
    y2 = data.y2_ff[8]
    r = y2/y1
    f,noise_v = integrated_rms_noise(x,r,normalize=False,freqs=np.linspace(1,130))
    noise_v = noise_v/25
    fig_integral,ax_integral=plt.subplots(2,1,figsize=[6,4],sharex=True)
    ax_integral[0].plot(f,noise_v*1e3)
    ax_integral[0].set_ylabel("integrated V\nRMS noise (µm)")

    


    data=read("2022_02_ccm_test/run026.h5",fmax=fmax)
    idx = [20,30,40]
    freq_analysis(data,scanpoints=idx,fmax=fmax,use_monoangle=False,asd_scale=6,xlim=50,time_y_scale=0.1)
    plt.savefig("run026_freq_analysis.pdf",transparent=True)
    analyze_stability_run(data,fmax=fmax,factor_std=100,use_monoangle=False)
    fig = plt.gcf()
    ax = fig.axes[1]
    for i in idx:
        ax.axvline(data.positions[i],color="0.5")
    plt.savefig("run026_stability.pdf",transparent=True)

    x = data.xdata
    y1 = data.y1_ff[30]
    y2 = data.y2_ff[30]
    r = y2/y1
    f,noise_h = integrated_rms_noise(x,r,normalize=False,freqs=np.linspace(1,130))
    noise_h = np.sqrt(noise_h**2-noise_h[0]**2) # remove baseline noise
    noise_h = noise_h/17.7
    noise_h = noise_h/noise_h[-1]*180e-6 # I know it must converge to 180nm

    ax_integral[1].plot(f,noise_h*1e6)
    ax_integral[1].set_xlabel("freq (Hz)")
    ax_integral[1].set_ylabel("integrated H\nRMS noise (nm)")
    for a in ax_integral: a.grid()

    fig_integral.tight_layout()
    fig_integral.savefig("run023_run026_integrated_noise.pdf",transparent=True)

def refill():
    d = read("2022_02_ccm_test/run018.h5")
    x = d.xdata
    y = d.y1[1]
    y = y/y[0:10000].mean()
    peaks = signal.find_peaks(-y,distance=100,width=100,prominence=0.3)

    fig,ax=plt.subplots(1,2,figsize=[12,6],sharey=True)

    for pos in peaks[0]:
        ax[0].axvline(x[pos],color="#00FA9A",lw=1.5)
    ax[0].plot(x,y)
    ax[0].set_xlabel("time from start of acq (s)")
    ax[0].set_ylabel("monochroatic intensity\n(normalized pd1 amplifier output)")

    t0 = x[peaks[0][0]]
    x = x-t0
    idx = (x>-0.01)&(x<0.03)
    ax[1].plot(x[idx]*1e3,y[idx])

    ax[1].set_xlabel("Δt (ms)")
    ax[0].set_ylim(0,None)
    plt.tight_layout()
    plt.savefig("run018_refill.pdf",transparent=True)
    for a in ax: a.grid()

def analyze_monot(data,diode="y1",scanpoints="auto",median_filter=5,freq_filt=False):
    colors = "#d7191c #fdae61 #ffffbf #abd9e9 #2c7bb6".split()
    colors = "#7fc97f #beaed4 #fdc086 #ffff99 #386cb0".split()
    x = data.xdata
    y = data.get(diode)
    if median_filter>0:
        y = np.asarray( [signal.medfilt(yi,median_filter) for yi in y] )
    pos = data.positions*13.5 # convert to urad (based on M. Wulff calib)
    freqs = [
            [190,220],
            [1150,1160],
            [1835,1850],
            [2100,2200],
            ]
    if freq_filt:
        y = np.asarray( [bandpass(x,yi,freq=freqs) for yi in y] )
    fig,ax=plt.subplots(4,1,figsize=[10,10])
    ym = y.mean(1)
    ax[0].set_title(data.filename)
    ax[0].plot(pos,ym)
    ax[0].set_ylabel("freq filtered average")

    ax[1].plot(pos,np.abs(np.gradient(ym,pos)))
    ax[1].set_ylabel("derivative of\nfreq filtered average")

    std = y.std(1)
    ax[2].plot(pos,std)
    ax[2].set_ylabel("RMS of\nfreq filtered data")
    ax[2].set_xlabel("monot_converted (urad)")
    if isinstance(scanpoints,str) and scanpoints == "auto":
        scanpoints = [np.argmax(std),np.argmax(ym)]
    for p,c in zip(scanpoints,colors):
        for a in ax[:3]: a.axvline(pos[p],color=c)
        ax[3].plot(*amplitude_spectral_density(x,data.get(diode)[p]),label=pos[p],alpha=0.9,color=c)
    plt.legend(title="monot")
    ax[3].set_xlabel("freq (Hz)")
    ax[3].set_ylabel("ASD")
    ax[3].set_xlim(0,2500)
    for a in ax: a.grid()
    plt.tight_layout()


def analyze_step(d,N=100,deg_to_trans=125):
    d.t=d.y2_ff/d.y1_ff
    av = average_every
    d.urad = np.deg2rad((d.t-d.t[:100].mean())/deg_to_trans)*1e6
    plt.plot(av(d.xdata,N),av(d.urad,N))
    plt.title(d.filename)
    plt.xlabel("time (s)")
    plt.ylabel("urad")
    plt.grid()
    plt.tight_layout()


