import datastorage
import numpy as np
from . import eos
from . import tds
from openlab.oscilloscopes import lecroy

def analyze_pyroelectric(traces,percentile=(15,85)):

    """
    analyze trace recorder with pyroelectric detector

         -----    ----
         |   |    |  |
    _____|   |____|  |___
    """
    data = np.percentile(traces,percentile,axis=-1)
    return data[1]-data[0]

def eosreadscan(fname,as_voltage=True):
    """
    read saved scans taking care of converting string to dictionary and ADU to V (if needed)
    Parameters
    ----------
    """
    d = datastorage.read(str(fname))
    try:
        d.y1info = lecroy.deserialize_descr(d.y1info[0])
        d.y2info = lecroy.deserialize_descr(d.y2info[0])
    except Exception as e:
        # might not be needed is saved as dictionary
        print("*** Failed to convert lecroy info; error was",e)
    if as_voltage and d.y1data.dtype in (np.int8,np.int16):
        y1gain,y1off = d.y1info["vertical_gain"],d.y1info["vertical_offset"]
        d.y1data = d.y1data*y1gain-y1off
        y2gain,y2off = d.y2info["vertical_gain"],d.y2info["vertical_offset"]
        d.y2data = d.y2data*y2gain-y2off
    return d

