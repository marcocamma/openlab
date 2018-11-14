import socket
import time
import struct
from collections import namedtuple
import numpy as np
from ..utils import connection_drivers

_header = namedtuple("header","nbytes,nbits,encoding,binary_format1,binary_format2,endianess,info,npoints_tot,pt_format,pt_order,xunit,dx,x0,pt_off,yunit,dy,yoff,y0,notknown1,notknown2")


def interpret_wfm_header(data):
#    s.scpi.ask("WFMOutpre",as_str=False)
    #:WFMOUTPRE:BYT_NR 1;BIT_NR 8;ENCDG ASCII;BN_FMT RI;BYT_OR MSB;WFID "D5, unknown coupling,100.0us/div, 10000 points, Digitalmode";NR_PT 25;PT_FMT Y;PT_ORDER LINEAR;XUNIT "s";XINCR 100.0000E-9;XZERO -500.0000E-6;PT_OFF 0;YUNIT "State";YMULT 1.0000;YOFF 0.0E+0;YZERO 0.0E+0#
    data = data.split(";")
    tofloat = "dx,x0,dy,y0,yoff".split(",")
    for temp in tofloat:
        i = _header._fields.index(temp)
        data[i] = float(data[i])

    toint = "npoints_tot,nbytes,nbits".split(",")
    for temp in toint:
        i = _header._fields.index(temp)
        data[i] = int(data[i])

    header = _header(*data)
    return header

class TektronixScope:
    def __init__(self,dev_address="/dev/usbtmc1",debug=False):
        connection_drivers.DEBUG = debug
        if isinstance(dev_address,str):
            self.scope = connection_drivers.USBTMC(dev_address)
        else:
            self.scope = connection_drivers.SocketConnection(*dev_address)
        self.scope.write(":HEADer OFF")
        self.scope.write("WFMOUTPRE:BIT_NR 16")
        self.scope.write("WFMOutpre:BN_Fmt RI")
        self.scope.write("WFMOutpre:ENCdg BINary")
        self.scope.write("WFMOutpre:BYT_Or LSB")

    def reset(self):
        self.scope.write("*RST")
        self.scope.write("*CLS")

    def get_ch(self,ch=1,segment="all",nseg=1000,units="V",return_time=False):
        scope = self.scope
        scope.write('DATA:SOU CH%d'%ch)
        header = scope.ask("WFMOutpre")
        #print(header)
        header = interpret_wfm_header(header)

        npoints_per_trace= int(scope.ask("horizontal:recordlength"))

        scope.write('CURVE?')
        data = scope.read_nbytes(2507*nseg)
        
        for i in range(2,100):
            try:
                int(data[1:i])
            except ValueError:
                break
        flag = data[:i-1]
        if flag.decode('ascii')[-1] != '0': flag = flag[:-1] # sometimes one character passes the test above
        data = data.split(flag)[1:] # first is empty
        data = [np.fromstring(d[:-1],dtype=np.int16) for d in data] # last char of each group is terminator
        if len(data) == 1:
            ADC_wave = data[0]
            nseg = 1
        else:
            ADC_wave = np.asarray(data)
            nseg = ADC_wave.shape[0]

        volts = (ADC_wave - header.yoff) * header.dy + header.y0

        if nseg > 1 and segment != 'all':
            volts = volts[segment]

        if return_time:
            t = np.arange(npoints_per_trace)*header.dx+header.x0
            return t,volts
        else:
            return volts
