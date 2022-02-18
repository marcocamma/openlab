import socket
import time
import struct
from collections import namedtuple
import numpy as np
import connection_drivers

def interpret_wfm_header(data):
#    s.scpi.ask("WFMOutpre",as_str=False)
    #:WFMOUTPRE:BYT_NR 1;BIT_NR 8;ENCDG ASCII;BN_FMT RI;BYT_OR MSB;WFID "D5, unknown coupling,100.0us/div, 10000 points, Digitalmode";NR_PT 25;PT_FMT Y;PT_ORDER LINEAR;XUNIT "s";XINCR 100.0000E-9;XZERO -500.0000E-6;PT_OFF 0;YUNIT "State";YMULT 1.0000;YOFF 0.0E+0;YZERO 0.0E+0#
    data_dict = dict()
    for s in data:
        key = s.lower().replace("wfmoutpre:","")
        data_dict[key]=data[s]
    tofloat = "xincr,xzero,ymult,yzero,yoff".split(",")
    for i in tofloat:
        data_dict[i] = float(data_dict[i])

    toint = "nr_pt,byt_nr,bit_nr,pt_off".split(",")
    for i in toint:
        data_dict[i] = int(data_dict[i])

#    header = _header(*data)
    return data_dict



class TektronixScope:
    def __init__(self,dev_address="/dev/usbtmc1",debug=False):
        connection_drivers.DEBUG = debug
        if isinstance(dev_address,str):
            self.scope = connection_drivers.USBTMC(dev_address)
        else:
            self.scope = connection_drivers.SocketConnection(*dev_address)
        self.scope.write(":HEADer ON")
        self.scope.write("WFMOUTPRE:BIT_NR 16")
        self.scope.write("WFMOutpre:BN_Fmt RI")
        self.scope.write("WFMOutpre:ENCdg BINary")
        self.scope.write("WFMOutpre:BYT_Or LSB")

    def write(self,cmd):
        self.scope.write(cmd)

    def ask(self,cmd,as_dict=True,remove_header=False,cast=None):

        if cast is not None: remove_header = True
        if remove_header: as_dict = False

        if not cmd.startswith(":"): cmd = ":" + cmd

        data = self.scope.ask(cmd)
        if as_dict:
            data = data.split(";")
            data_dict = dict()
            for s in data:
                key,value = s.split(" ",maxsplit=1)
                key = key.lstrip(":")
                data_dict[key]=value.lstrip()
            return data_dict
        else:
            if remove_header:data = data.replace(cmd.upper(),"").lstrip()
            if cast: data = cast(data)
            return data


    def is_fastframe_sumframe(self):
        return self.ask("HORizontal:FASTframe:SUMFrame:STATE",remove_header=True,cast=bool)

    def get_numfastframes(self):
        return self.ask("ACQuire:NUMFRAMESACQuired",remove_header=True,cast=int)

    def reset(self):
        self.write("*RST")
        self.write("*CLS")
        self.write("HEAD ON")

    def get_header(self,ch=1):
        self.write('DATA:SOU CH%d'%ch)
        header = self.ask("WFMOutpre",as_dict=True)
        header = interpret_wfm_header(header)
        return header

    def get_xaxis(self):
        header = self.get_header()
        npoints_per_trace = header["nr_pt"]
        t = (np.arange(npoints_per_trace)-header["pt_off"])*header['xincr']+header['xzero']
        return t

    def acquire_mode(self,mode="HIRes",single_mode=True):
        modes =  "sample","peakdetect","hires","average","envelope"
        if not mode.lower() in modes:
            raise ValueError("mode has to be one of",modes)
        else:
            self.write("ACQuire:MODe "+mode)

    def enable_burst(self):
        self.write('ACQuire:STOPAfter SEQUENCE')

    def disable_burst(self):
        self.write('ACQuire:STOPAfter RUNSTop')

    def set_num_acq(self, n):
        self.write('ACQUIRE:SEQuence:NUMSEQuence '+str(n))

    def get_num_acq(self):
        return self.ask('ACQUIRE:NUMACQ',cast=int)

    def is_acquiring(self):
        return self.ask('ACQuire')['STATE']=='1'
    
    def wait_for_end(self,poll=0.1):
        while self.is_acquiring():
            time.sleep(poll)


    def acquisition_start(self):
        self.write("ACQuire:STATE RUN")

    def acquisition_stop(self):
        self.write("ACQuire:STATE STOP")

    def trigger_mode(self,mode="normal"):
        modes = "normal","auto"
        if mode.lower() not in modes:
            raise ValueError("mode has to be one of",modes)
        self.write("TRIGger:A:MODe "+mode)

    def prepare_fastframe(self,nframes=10,start=False):
        self.acquisition_stop()
        self.write("HORizontal:FASTframe:STATE ON")
        self.set_num_acq(1) # each will be made of nframes
        self.write("HORizontal:FASTframe:COUNt %d"%nframes)
        self.enable_burst()
        self.trigger_mode(mode="normal")
        if start:
            self.acquisition_start()

    def acquire_single_trigger(self):
        self.set_num_acq(1)
        self.acquisition_start()

    def get_ch(self,ch=1,units="V",return_time=False):
        """
        this function does not return the summary_frame even if acquired
        if precise_timing: read sub sample offset for each fastframe trace, in this case t will be a [nframes,npoints] 2D array
        """
        scope = self.scope
        scope.write('DATA:SOU CH%d'%ch)

        header = self.get_header(ch=ch)

        # print(header)

        nframes = 1 # it does not include 'summary' frame

        npoints_per_trace = header["nr_pt"]
        t = (np.arange(npoints_per_trace)-header["pt_off"])*header['xincr']+header['xzero']

        bytes_per_point = header["byt_nr"]
        npoints_per_trace = header["nr_pt"]

        nbytes_data_per_trace = npoints_per_trace*bytes_per_point
        # read binary header
        # <Block> is the waveform data in binary format. The waveform is formatted as:
        # #<x><yyy><data><newline>, where: <x> is the number of y bytes. For example, if <yyy>=500, then <x>=3

        yyy = str(nbytes_data_per_trace)
        x = len(yyy)

        preamble = ":CURVE #"+str(x)+str(yyy)
        nbytes_per_trace = len(preamble)+nbytes_data_per_trace+1 # +1 is for terminator
        nbytes = nbytes_per_trace

        t0 = time.time()
        scope.write("CURVE?")
        data = scope.read_nbytes_preallocate(nbytes)
        t1 = time.time()
        #print(f"to read time {t1-t0:.3f}s; data_len {len(data)}")

        t0 = time.time()
        temp = []
        start = len(preamble)
        stop  = start + nbytes_data_per_trace
        ADC_wave = np.frombuffer(data[start:stop],dtype=np.int16)
        t1 = time.time()
        #print(f"to convert time {t1-t0:.3f}s")

        if units == "V":
            ret = (ADC_wave - header['yzero']) * header['ymult'] + header['yzero']
        else:
            ret = ADC_wave

        if return_time:
            return t,ret
        else:
            return ret

    def get_ch_fastframe(self,ch=1,segment="all",units="V",return_time=False,precise_timing=False):
        """
        this function does not return the summary_frame even if acquired
        if precise_timing: read sub sample offset for each fastframe trace, in this case t will be a [nframes,npoints] 2D array
        """
        scope = self.scope
        scope.write('DATA:SOU CH%d'%ch)

        header = self.get_header(ch=ch)

        # print(header)

        nframes = self.get_numfastframes() # it does not include 'summary' frame
        with_summary_frame = self.is_fastframe_sumframe()

        npoints_per_trace = header["nr_pt"]
        if nframes == 1 or not precise_timing:
            t = (np.arange(npoints_per_trace)-header["pt_off"])*header['xincr']+header['xzero']
        else:
            offsets = self.ask("HORizontal:FASTframe:XZEro:ALL",remove_header=True)
            # ['"1: 1.3125E-11"', '"2: 4.5312E-12"', '"3: 1.8281E-11"']
            offsets = [float(o.split(":")[1][:-1]) for o in offsets.split(",")]

            t = [(np.arange(npoints_per_trace)-header["pt_off"])*header['xincr']+offsets[i] for i in range(nframes)]
            t = np.asarray(t)

        if with_summary_frame:
            summary_frame = 1
        else:
            summary_frame = 0

#        npoints_per_trace= int(scope.ask("horizontal:recordlength",encoding='utf8'))

        bytes_per_point = header["byt_nr"]
        npoints_per_trace = header["nr_pt"]

        nbytes_data_per_trace = npoints_per_trace*bytes_per_point
        # read binary header
        # <Block> is the waveform data in binary format. The waveform is formatted as:
        # #<x><yyy><data><newline>, where: <x> is the number of y bytes. For example, if <yyy>=500, then <x>=3

        yyy = str(nbytes_data_per_trace)
        x = len(yyy)

        preamble = ":CURVE #"+str(x)+str(yyy)
        nbytes_per_trace = len(preamble)+nbytes_data_per_trace+1 # +1 is for terminator
        nbytes = nbytes_per_trace*(nframes+summary_frame)

        t0 = time.time()
        scope.write("CURVE?")
        data = scope.read_nbytes_preallocate(nbytes)
        t1 = time.time()
        #print(f"to read time {t1-t0:.3f}s; data_len {len(data)}")

#        t0 = time.time()
#        scope.write("CURVE?")
#        data = scope.read_nbytes(nbytes)
#        t1 = time.time()
#        print(f"to read time {t1-t0:.3f}s")
       
        t0 = time.time()
        temp = []
        for i in range(nframes):
            start = nbytes_per_trace*i+len(preamble)
            stop  = start + nbytes_data_per_trace
            temp.append( np.frombuffer(data[start:stop],dtype=np.int16) )
        t1 = time.time()
        #print(f"to convert time {t1-t0:.3f}s")
        if len(temp) == 1:
            ADC_wave = temp[0]
            nseg = 1
        else:
            ADC_wave = np.asarray(temp)
            nseg = ADC_wave.shape[0]

        if units == "V":
            ret = (ADC_wave - header['yzero']) * header['ymult'] + header['yzero']
        else:
            ret = ADC_wave

        if nseg > 1 and segment != 'all':
            ret = ret[segment]

        if return_time:
            return t,ret
        else:
            return ret
