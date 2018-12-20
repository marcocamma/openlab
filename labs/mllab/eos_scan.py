import numpy as np
import time

from . import thzsetup
from . import config
import pathlib

storage = config.storage
data_folder = pathlib.Path("/data/thz/setup/2018.06")

ds = thzsetup.delay_stage
scope = thzsetup.scope

def acquire(nshots=100,use_single=True):
    x = scope.get_xaxis()
    y1 = np.zeros( (nshots,len(x)) )
    y2 = np.zeros( (nshots,len(x)) )
    for i in range(nshots):
        if use_single: scope.trigger("SINGLE")
        y1[i] = scope.get_waveform(1)[1]
        y2[i] = scope.get_waveform(2)[1]
    return x,y1,y2


def scan(first,last,step=0.1,shots=100,repeat=3):
    run = storage.get("RUN_NUMBER",0)+1
    storage['RUN_NUMBER'] = run
    storage.save()
    delay_list = np.arange(first,last+step,step)
    last_y1 = np.arange(10)
    for nrep in range(repeat):
        for idelay,delay in enumerate(delay_list):
            ds.move(delay); ds.wait()
            x,y1,y2 = acquire(shots)
            if last_y1 == y1:
                print("Warning, last two waveforms1 are the same, laser down ?")
                last_y1 = y1
            fname = data_folder / str("run%04d/run%04d_pos%05d_rep%02d" % 
                    (run,run,idelay,nrep))
            fname.parent.mkdir(parents=True,exist_ok=True)
            np.savez(str(fname),x=x,y1=y1,y2=y2,delay=delay,
                    stage=ds._motor.wmd())
            delay = "%.3f" % delay
            print(time.asctime(),run,idelay,len(delay_list),nrep,repeat,delay)


def find(start=-100,N=1000,dt=0.5,pause=0.5):
    ds.move(start)
    for i in range(N):
        ds.mvr(dt)
        ds.wait()
        print(ds.wm())
        time.sleep(pause)

