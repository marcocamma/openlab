""" 
read .trc lecroy data,
inspired by M Betz work
"""
import datetime
import numpy as np
import struct
import collections

lecroy_trace = collections.namedtuple("lecroy_trace", "x y info")


# template to read LECROY_2_3
# based on (http://forums.ni.com/attachments/ni/60/4652/2/LeCroyWaveformTemplate_2_3.pdf)
# format of each entry if offset,datatype
_template = dict(
    lWAVE_DESCRIPTOR=(36, "l"),
    lUSER_TEXT=(40, "l"),
    lTRIGTIME_ARRAY=(48, "l"),
    lRIS_TIME_ARRAY=(52, "l"),
    lWAVE_ARRAY_1=(60, "l"),
    lWAVE_ARRAY_2=(64, "l"),
    INSTRUMENT_NAME=(76, "16s"),
    INSTRUMENT_NUMBER=(92, "l"),
    TRACE_LABEL=(96, "16s"),
    WAVE_ARRAY_COUNT=(116, "l"),
    PNTS_PER_SCREEN=(120, "l"),
    FIRST_VALID_PNT=(124, "l"),
    LAST_VALID_PNT=(128, "l"),
    FIRST_POINT=(132, "l"),
    SPARSING_FACTOR=(136, "l"),
    SEGMENT_INDEX=(140, "l"),
    SUBARRAY_COUNT=(144, "l"),
    SWEEPS_PER_ACQ=(148, "l"),
    POINTS_PER_PAIR=(152, "h"),
    PAIR_OFFSET=(154, "h"),
    VERTICAL_GAIN=(156, "f"),
    VERTICAL_OFFSET=(160, "f"),
    MAX_VALUE=(164, "f"),
    MIN_VALUE=(168, "f"),
    NOMINAL_BITS=(172, "h"),
    NOM_SUBARRAY_COUNT=(174, "h"),
    HORIZ_INTERVAL=(176, "f"),
    HORIZ_OFFSET=(180, "d"),
    PIXEL_OFFSET=(188, "d"),
    VERTUNIT=(196, "48s"),
    HORUNIT=(244, "48s"),
    HORIZ_UNCERTAINTY=(292, "f"),
    ACQ_DURATION=(312, "f"),
    RECORD_TYPE=(316, "H"),
    PROCESSING_DONE=(318, "H"),
    RIS_SWEEPS=(322, "h"),
    TIMEBASE=(324, "H"),
    VERT_COUPLING=(326, "H"),
    PROBE_ATT=(328, "f"),
    FIXED_VERT_GAIN=(332, "H"),
    BANDWIDTH_LIMIT=(334, "H"),
    VERTICAL_VERNIER=(336, "f"),
    ACQ_VERT_OFFSET=(340, "f"),
    WAVE_SOURCE=(344, "H"),
)


_mapping = dict(
    RECORD_TYPE=[
        "single_sweep",
        "interleaved",
        "histogram",
        "graph",
        "filter_coefficient",
        "complex",
        "extrema",
        "sequence_obsolete",
        "centered_RIS",
        "peak_detect",
    ],
    PROCESSING_DONE=[
        "no_processing",
        "fir_filter",
        "interpolated",
        "sparsed",
        "autoscaled",
        "no_result",
        "rolling",
        "cumulative",
    ],
    TIMEBASE=[
        "1_ps/div",
        "2_ps/div",
        "5_ps/div",
        "10_ps/div",
        "20_ps/div",
        "50_ps/div",
        "100_ps/div",
        "200_ps/div",
        "500_ps/div",
        "1_ns/div",
        "2_ns/div",
        "5_ns/div",
        "10_ns/div",
        "20_ns/div",
        "50_ns/div",
        "100_ns/div",
        "200_ns/div",
        "500_ns/div",
        "1_us/div",
        "2_us/div",
        "5_us/div",
        "10_us/div",
        "20_us/div",
        "50_us/div",
        "100_us/div",
        "200_us/div",
        "500_us/div",
        "1_ms/div",
        "2_ms/div",
        "5_ms/div",
        "10_ms/div",
        "20_ms/div",
        "50_ms/div",
        "100_ms/div",
        "200_ms/div",
        "500_ms/div",
        "1_s/div",
        "2_s/div",
        "5_s/div",
        "10_s/div",
        "20_s/div",
        "50_s/div",
        "100_s/div",
        "200_s/div",
        "500_s/div",
        "1_ks/div",
        "2_ks/div",
        "5_ks/div",
        "EXTERNAL",
    ],
    VERT_COUPLING=["DC_50_Ohms", "ground", "DC_1MOhm", "ground", "AC,_1MOhm"],
    FIXED_VERT_GAIN=[
        "1_uV/div",
        "2_uV/div",
        "5_uV/div",
        "10_uV/div",
        "20_uV/div",
        "50_uV/div",
        "100_uV/div",
        "200_uV/div",
        "500_uV/div",
        "1_mV/div",
        "2_mV/div",
        "5_mV/div",
        "10_mV/div",
        "20_mV/div",
        "50_mV/div",
        "100_mV/div",
        "200_mV/div",
        "500_mV/div",
        "1_V/div",
        "2_V/div",
        "5_V/div",
        "10_V/div",
        "20_V/div",
        "50_V/div",
        "100_V/div",
        "200_V/div",
        "500_V/div",
        "1_kV/div",
    ],
    BANDWIDTH_LIMIT=["off", "on"],
)


def read_trc(fName, as_volt=True):
    """
        Reads .trc binary files from LeCroy Oscilloscopes.
       
        Parameters
        -----------       
        fName = filename of the .trc file
        as_volt = bool, if false return ADC counts
        
        Returns
        -----------
        namedtuple (x,y,info)
    """
    with open(fName, "rb") as fid:

        data = fid.read(50).decode()
        offset = data.find("WAVEDESC")

        # get endianess
        endianess = "<" if get(fid, "H", offset + 34) else ">"

        info = dict()
        for key, (position, fmt) in _template.items():
            if fmt.find("s") < 0:
                fmt = endianess + fmt
            info[key] = get(fid, fmt, position + offset)

        string_keys = [
            key for key, value in _template.items() if value[1].find("s") > 0
        ]

        for key in string_keys:
            info[key] = info[key].decode().split("\x00")[0]

        for key, mapping in _mapping.items():
            info[key] = mapping[info[key]]

        info["TRIGGER_TIME"] = get_time_stamp(fid, endianess, offset + 296)

        # read user text
        fmt = "{0}s".format(info["lUSER_TEXT"])
        off = offset + info["lWAVE_DESCRIPTOR"]
        info["USER_TEXT"] = get(fid, fmt, off).decode().split("\x00")[0]

        # read data
        fmt = "int16" if get(fid, "H", offset + 32) else "int8"
        off = (
            offset
            + info["lWAVE_DESCRIPTOR"]
            + info["lUSER_TEXT"]
            + info["lTRIGTIME_ARRAY"]
            + info["lRIS_TIME_ARRAY"]
        )
        fid.seek(off)
        y = np.fromfile(fid, fmt, info["lWAVE_ARRAY_1"])
        if endianess == ">":
            y.byteswap(True)
        if as_volt:
            y = info["VERTICAL_GAIN"] * y - info["VERTICAL_OFFSET"]
        x = np.arange(1, len(y) + 1) * info["HORIZ_INTERVAL"] + info["HORIZ_OFFSET"]
    return lecroy_trace(x=x, y=y, info=info)


def get(fid, fmt, adr=None):
    """ extract a byte / word / float / double from the binary file """
    nBytes = struct.calcsize(fmt)
    if adr is not None: fid.seek(adr)
    s = struct.unpack(fmt, fid.read(nBytes))
    if type(s) == tuple:
        return s[0]
    else:
        return s


def get_time_stamp(fid, endi, adr):
    """ extract a timestamp from the binary file """
    s = get(fid, endi + "d", adr)
    m = get(fid, endi + "b")
    h = get(fid, endi + "b")
    D = get(fid, endi + "b")
    M = get(fid, endi + "b")
    Y = get(fid, endi + "h")
    trigTs = datetime.datetime(Y, M, D, h, m, int(s), int((s - int(s)) * 1e6))
    return trigTs
