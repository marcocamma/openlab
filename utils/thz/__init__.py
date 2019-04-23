from . import eos
from . import tds

def analyze_pyroelectric(traces,percentile=(15,85)):

    """
    analyze trace recorder with pyroelectric detector

         -----    ----
         |   |    |  |
    _____|   |____|  |___
    """
    data = np.percentile(traces,percentile,axis=-1)
    return data[1]-data[0]

