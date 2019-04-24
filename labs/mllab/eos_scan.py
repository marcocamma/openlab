import numpy as np
import time

from . import thzsetup
from . import config
from openlab.generic import scan
import pathlib
from openlab.oscilloscopes import lecroy
import datastorage
from datastorage import DataStorage

storage = config.storage

delay_stage = thzsetup.delay_stage
scope = thzsetup.scope
scope.get_16bits()

def acquire(nshots=100,use_single=True):
    scope.set_sequence_mode(nshots)
    scope.trigger("SINGLE")
    y1 = scope.get_waveform(1,units="ADU",serialize=True)
    y1info,y1data = y1
    y2 = scope.get_waveform(2,units="ADU",serialize=True)
    y2info,y2data = y2
    x = scope.get_xaxis()[:y2data.shape[1]]
    return dict( _xdata=x,
            y1info=y1info,y1data=y1data,
            y2info=y2info,y2data=y2data,
            delay_stage_mm_dial=delay_stage.parents.wmd(),
            delay_stage_mm=delay_stage.parents.wm()
            )


def eosreadscan(fname,as_voltage=True,folder="auto"):
    """
    read saved scans taking care of converting string to dictionary and ADU to V (if needed)
    Parameters
    ----------
    fname : filepath or runnumber
        if runnumber, the folder used is defined by the 'folder' keyword
    folder : path or "auto"
        if auto uses variable defined in global storage (EOS_FOLDER field)
        this field is used only if fname is a run numer
    """
    if isinstance(fname,int):
        if folder == "auto":
            if not "EOS_FOLDER" in storage:
                raise RuntimeError("EOS_FOLDER not defined global storage, can't continue, please add or specify folder")
            else:
                folder = pathlib.Path(storage["EOS_FOLDER"])
        else:
            folder = pathlib.Path(folder)
        fname = folder / str("scan%04d.h5"%fname)

    d = datastorage.read(str(fname))
    try:
        d.y1info = [lecroy.deserialize_descr(info) for info in d.y1info]
        d.y2info = [lecroy.deserialize_descr(info) for info in d.y2info]
    except:
        # might not be needed is saved as dictionary
        pass
    if as_voltage and d.y1data.dtype in (np.int8,np.int16):
        y1gain,y1off = d.y1info[0]["vertical_gain"],d.y1info[0]["vertical_offset"]
        d.y1data = d.y1data*y1gain-y1off
        y2gain,y2off = d.y2info[0]["vertical_gain"],d.y2info[0]["vertical_offset"]
        d.y2data = d.y2data*y2gain-y2off
    return d



def eosscan(first,last,step=0.1,nshots=100,folder="auto",comment=None):
    """
    Parameters
    ----------
    folder : path or "auto"
        if auto uses variable defined in global storage (EOS_FOLDER field)
    """
    if folder == "auto":
        if not "EOS_FOLDER" in storage:
            raise RuntimeError("EOS_FOLDER not defined global storage, can't continue, please add or specify folder")
        else:
            folder = pathlib.Path(storage["EOS_FOLDER"])
    else:
        folder = pathlib.Path(folder)

    run = storage.get("EOS_RUN_NUMBER",0)+1
    storage['EOS_RUN_NUMBER'] = run
    storage.save()
    fname = folder / str("scan%04d.h5"%run)
    N = int((last-first)/step)
    def _acq():
        return acquire(nshots=nshots)
    _acq(); # do an acquisition to be sure things are ready to go
    scan.ascan(delay_stage,first,last,N,acquire=_acq,fname=fname,comment=comment)


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
            data['stage'] = delay_stage.parents.wmd()
            data['delay'] = delay_stage.wm()
            fname = data_folder / str("scan%04d/scan%04d_pos%05d_rep%02d.h5" %
                    (run,run,idelay,nrep))
            fname.parent.mkdir(parents=True,exist_ok=True)
            data.save(str(fname))
            delay = "%.3f" % delay
            print(time.asctime(),run,idelay,len(delay_list),nrep,repeat,delay)
