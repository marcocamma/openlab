import socket
import time
import struct
from collections import namedtuple
import numpy as np


class SCPI:
    PORT = 5025

    def __init__(self, host="10.1.1.2", port=PORT):
        self.host = host
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))

        self.f = self.s.makefile("rb")

    def send(self,msg):
        if not msg.endswith("\n"): msg += "\n"
        self.s.send(msg.encode("ascii"))

    def recv(self,as_str=True):
        """
        Return a message from the scope.
        """
        reply = b""
        while not reply.endswith("\n".encode("ascii")):
            reply += self.s.recv(2**20)
        if as_str:
            reply = reply.decode("ascii")
            reply = reply.strip()
        return reply

    def query(self,msg,as_str=True):
        if not msg.endswith("?"): msg += "?"
        self.send(msg)
        return self.recv(as_str=as_str)


preamble = namedtuple("preamble","format,type,npoints,avg_count,dx,x0,xref,dy,y0,yref,coupling,xdisp_range,xdisp_orig,ydisp_range,ydisp_orig,date,time,model,acqmode,completion,xunits,yunits,bandwidth_max,bandwidth_min")
preamble_seg = namedtuple("preamble",list(preamble._fields) + ["segment_number",])

_acqmode = { 0 : "RTIM", 1 : "ETIM", 2 : "SEGM", 3 : "PDET" }

def interpret_preamble(data):
    data = data.split(",")

    tofloat = "dx,x0,xref,dy,y0,xdisp_range,xdisp_orig,ydisp_range,ydisp_orig,yunits,bandwidth_max,bandwidth_min".split(",")
    for temp in tofloat:
        i = preamble._fields.index(temp)
        data[i] = float(data[i])

    toint = "npoints,avg_count".split(",")
    for temp in toint:
        i = preamble._fields.index(temp)
        data[i] = int(data[i])


    i = preamble._fields.index("acqmode")
    data[i] = _acqmode[int(data[i])]

    if len(data) == 24:
        data = preamble(*data)
    else:
        data = preamble_seg(*data)
    return data

cmd = namedtuple("command","scpi,info,type")
keysight_databse = dict(
        streaming = cmd(":WAV:STR","use streaming",""),
        acq_mode = cmd(":ACQ:MODE","acquisition mode",("RTIM","SEQ")),
        info     = cmd(":WAV:PRE","waveform preamble",""),
        wave_format = cmd(":WAV:FORM","format to transfer data",("ASCII","WORD")),
        trigger_holdoff = cmd(":TRIGger:HOLDoff","scope will not trigger before this time has elapsed",float),
        preamble = cmd(":WAVeform:PREamble","info on waveform","")
)

class KeysightScope:
    def __init__(self,host="10.1.1.2",port=5025):
        print("Test if noise comes from long holdofftime (should not)")
        self.scpi = SCPI(host=host,port=port)
        self.set("streaming",1)
        self.scpi.send(":SYST:HEAD OFF")
        self.scpi.send(":WAVeform:BYTeorder LSBFirst")
        self.scpi.send(":WAVeform:TYPE RAW")
        self.scpi.send(":ACQuire:INTerpolate OFF")
        self.set("wave_format","WORD")
        self.set("trigger_holdoff","1e-3")

    def reset(self):
        self.scpi.send("*RST")
        self.scpi.send("*CLS")

    def set(self,what,value,verify=True):
        cmd = keysight_databse[what]
        msg = cmd.scpi
        if verify and isinstance(cmd.type,(list,tuple)):
            assert value in cmd.type, "The value must be one of %s"%cmd.type
        msg = msg + " %s" % value
        self.scpi.send(msg)

    def get(self,what,as_str=True):
        cmd = keysight_databse[what]
        data = self.scpi.query(cmd.scpi,as_str=as_str)
        if callable(cmd.type):
            data = cmd.type(data)
        return data

    def list(self):
        cmd_list = list(keysight_databse.keys())
        cmd_list.sort()
        for c in cmd_list:
            print("%20s| %20s | %s"%(c,keysight_databse[c].scpi,keysight_databse[c].info))

    def get_ch(self,ch=1,segment="all",units="V",return_time=False):
        self.scpi.send(":WAV:SOUR CHAN%d"%ch)
        header = interpret_preamble(self.get("preamble"))
        if segment == "all":
            self.scpi.send(":WAVeform:SEGMented:ALL ON")
        else:
            self.scpi.send(":ACQuire:SEGMented:INDex %d"%segment)
        data = self.scpi.query("WAV:DATA",as_str=False)
        # bytes 0,1 and last are header
        data = np.frombuffer(data[2:-1], dtype=np.int16)
        if header.acqmode == "SEGM":
            data = data.reshape( (-1,header.npoints) )
        if units == "V": data = data*header.dy+header.y0
        if return_time:
            t = np.arange(header.npoints)*header.dx+header.x0
            return t,data
        else:
            return data
