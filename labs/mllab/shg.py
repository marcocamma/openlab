import numpy as np
import time

from . import thzsetup
from . import config
import pathlib
from openlab.oscilloscopes.lecroy import deserialize_descr
from datastorage import DataStorage

storage = config.storage
storage["SHG_FOLDER"] = "/data/thz/2019.04/SHG/"

scope = thzsetup.scope
if scope is not None: scope.get_16bits()


def convert_to_V(data):
    ydata = data["ydata"]
    info = deserialize_descr(data["yinfo"])
    if str(ydata.dtype) == "int16":
        ydata = ydata*info["vertical_gain"]-info["vertical_offset"]
        data["ydata"] = ydata
    return data
 

def analyze(data):
    data = convert_to_V(data) # convert from ADU to V if needed
    t = data["xdata"]*1e9
    data = data["ydata"]
    bkg = data[:,:30].mean(1)
    data = data-bkg[:,np.newaxis]
    idx = (t>0)&(t<50)
    counts = data[:,idx].sum(1)
    return counts




def _acquire_pmt(nshots=100,use_single=True):
    scope.set_sequence_mode(nshots)
    scope.trigger("SINGLE")
    time.sleep(0.1)
    while scope.query("TRMD?").strip() != "TRMD STOP":
        time.sleep(0.05)
    y = scope.get_waveform(4,units="ADU",serialize=True)
    yinfo,ydata = y
    x = scope.get_xaxis()[:ydata.shape[1]]
    return dict( xdata=x,
            yinfo=yinfo,ydata=ydata,
            )

def acquire_pmt(nshots=100,folder="auto",comment=None,save=True):
    """
    Parameters
    ----------
    folder : path or "auto"
        if auto uses variable defined in global storage (EOS_FOLDER field)
    """
    data = _acquire_pmt(nshots=nshots)
    data["counts"] = analyze(data)
    if comment is not None: data["comment"] = comment
    data = DataStorage(data)

    if save:
        if folder == "auto":
            if not "SHG_FOLDER" in storage:
                raise RuntimeError("SHG_FOLDER not defined global storage, can't continue, please add or specify folder")
            else:
                folder = pathlib.Path(storage["SHG_FOLDER"])
        else:
            folder = pathlib.Path(folder)

        run = storage.get("SHG_RUN_NUMBER",0)+1
        storage['SHG_RUN_NUMBER'] = run
        storage.save()
        fname = folder / str("shg_acq%04d.h5"%run)
        print("Saving in",str(fname))

        data.save(str(fname))
    return data



