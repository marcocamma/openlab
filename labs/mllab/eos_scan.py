import numpy as np
import time

from . import thzsetup
from . import config
from openlab.generic import scan
import pathlib

from datastorage import DataStorage

storage = config.storage

delay_stage = thzsetup.delay_stage
scope = thzsetup.scope
scope.get_16bits()

def acquire(nshots=100,use_single=True):
    scope.set_sequence_mode(nshots)
    scope.trigger("SINGLE")
    y1 = scope.get_waveform(1,units="ADU")
    y1info,y1data = y1
    y2 = scope.get_waveform(2,units="ADU")
    y2info,y2data = y2
    x = scope.get_xaxis()[:y2data.shape[1]]
    return dict( _xdata=x,
            y1info=y1info,y1data=y1data,
            y2info=y2info,y2data=y2data,
            delay_stage_mm_dial=delay_stage._motor.wmd(),
            delay_stage_mm=delay_stage._motor.wm()
            )


def eosscan(first,last,step=0.1,nshots=100):
    data_folder = pathlib.Path("/data/thz/2019.04/eos")
    run = storage.get("EOS_RUN_NUMBER",0)+1
    storage['EOS_RUN_NUMBER'] = run
    storage.save()
    fname = data_folder / str("scan%04d.h5"%run)
    N = int((last-first)/step)
    def _acq():
        return acquire(nshots=nshots)
    scan.ascan(delay_stage,first,last,N,acquire=_acq,fname=fname)


def eosscan_old(first,last,step=0.1,shots=100,repeat=3):
    data_folder = pathlib.Path("/data/thz_nosync/2019.04/eos")
    run = storage.get("RUN_NUMBER",0)+1
    storage['RUN_NUMBER'] = run
    storage.save()
    delay_list = np.arange(first,last+step,step)
    for nrep in range(repeat):
        for idelay,delay in enumerate(delay_list):
            delay_stage.move(delay); ds.wait()
            data = acquire(shots)
            data['stage'] = delay_stage._motor.wmd()
            data['delay'] = delay_stage.wm()
            fname = data_folder / str("scan%04d/scan%04d_pos%05d_rep%02d.h5" %
                    (run,run,idelay,nrep))
            fname.parent.mkdir(parents=True,exist_ok=True)
            data.save(str(fname))
            delay = "%.3f" % delay
            print(time.asctime(),run,idelay,len(delay_list),nrep,repeat,delay)
