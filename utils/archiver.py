"""
Read data sent via zmq format should be dict like object

data = { "time" : datetime.datetime.now(),
         "sensor1/value1" : 1,
         "sensor1/value2" : 2,
         "sensor2/value1" : 3,
         "sensor2/value2" : 4,
         }
"""
import zmq
import time
import datetime
import pathlib
import threading
import random
import pandas as pd # to average over time
import numpy as np

readers = {}

def now(): return datetime.datetime.now()


class Reader(object):
    def __init__(self,ip="127.0.0.1",port=1234,fname="auto",
            log_every="1T",fmt="%.2f",every=5,folder="./log",autostart=False,
            log_size_MB=10):
        """
        Parameters
        ----------
        log_every : string
          5s, 6m, etc.
        every: float
          minimum time (in sec) between reads
        """
        context = zmq.Context()
        sock = context.socket(zmq.SUB)
        addr = "tcp://%s:%s"%(ip,port)
        sock.connect(addr)
        sock.setsockopt(zmq.SUBSCRIBE,b'')
        self.sock = sock
        self.addr = addr
        self._stop = False
        self.done = False
        self.every = every
        if fname  == "auto": fname = "%s:%s"%(ip,port)
        self.fname = fname

        _units = log_every[-1]
        _value = float(log_every[:-1])
        if _units == "s":
            log_every = datetime.timedelta( seconds = _value)
        elif _units == "T":
            log_every = datetime.timedelta( minutes = _value)
        elif _units == "m":
            log_every = datetime.timedelta( minutes = _value)
        elif _units == "h":
            log_every = datetime.timedelta( hours = _value)
        else:
            log_every = datetime.timedelta( seconds = 0.1)
        self.log_every = log_every
        self.folder = folder
        self.fmt = fmt
        self.thread = None
        self.last_read = None
        self.time_to_saving = self.log_every
        self.last_logfname = self.set_new_filename()
        self.log_size_MB = log_size_MB
        if autostart: self.start()

    def set_new_filename(self):
        _ = now()
        self.last_logfname = self.fname + "_%04d%02d%02d_%02d%02d%02d.h5"%(_.year,_.month,_.day,_.hour,_.minute,_.second)
        return self.last_logfname

    def _save(self,datastorage):

        f = pathlib.Path(self.folder) / pathlib.Path(self.last_logfname)
        if not f.exists():
            f.parent.mkdir(parents=True,exist_ok=True)

        datastorage.resample(self.log_every,on="time").mean().to_hdf(str(f),key="/data")

        timev = datastorage['time']
        mem = self.datastorage.memory_usage(deep=True).sum() # in bytes
        is_log_too_big = mem > self.log_size_MB*1024*1024
        is_month_changed = timev[0].month != timev[len(timev)-1].month
        if is_log_too_big or is_month_changed:
            self.last_logfname = self.set_new_filename()
            # reset
            datastorage = pd.DataFrame()
        return datastorage


    def _start(self):
        self.done = False # in case we restart
        self._stop = False
        self.last_saved = now()

        # read if file exists ... 
        f = pathlib.Path(self.folder) / pathlib.Path(self.last_logfname)
        if not f.exists():
            datastorage = pd.DataFrame()
        else: # append
            datastorage = pd.read_hdf(str(f))

        while not self._stop:
            #print("waiting for data")
            try:
                data = self.sock.recv_pyobj()
                self.last_read = data
            except:
                print("Reading failed for",self.addr)
                pass
            datastorage = datastorage.append(data,ignore_index=True)
            self.time_to_saving = self.log_every - (data['time']-self.last_saved)
            if self.time_to_saving.total_seconds() < 0:
                datastorage=self._save(datastorage)
                self.last_saved = data["time"]
            self.datastorage = datastorage
        else:
            self.done = True

    def __repr__(self):
        s = "data reader on %s"%self.addr
        s += "\nfolder : %s"%self.folder
        s += "\ncurrent filename in use : %s"%self.last_logfname
        s += "\ntime to next save %.1fs" % self.time_to_saving.total_seconds()
        if self.last_read is not None:
            s += "\nnum of readings %d" % len(self.datastorage)

            mem = self.datastorage.memory_usage(deep=True).sum() # in bytes
            frac = mem/1014/1024/self.log_size_MB*100
            if np.log10(mem) >= 3 and np.log10(mem) < 6:
                mem = "%.1f Kbytes" % (mem/1024)
            elif np.log10(mem) >= 6 and np.log10(mem) < 9:
                mem = "%.1f Mbytes" % (mem/1024/1024)
            elif np.log10(mem) >= 9 and np.log10(mem) < 12:
                mem = "%.1f Gbytes" % (mem/1024/1024/1024)
            else:
                mem = "%d bytes"%mem


            s += "\nmemory usage %s (%.1f%% of max logsize)"%(mem,frac)

            t = str(self.last_read["time"])
            s += "\nlast read on %s" % t
            for k,v in self.last_read.items():
                if k != "time":
                    s += "\n%15s %s"%(k,v)
        else:
            s += "\nno data read until now"
        return s

    def start(self):
        t=threading.Thread(target=self._start)
        t.start()
        self.thread = t
        global readers
        readers[self.addr] = self

    def stop(self,wait=True):
        self._stop = True
        if wait:
            while not self.done: time.sleep(0.01)



def stop_readers():
    global readers
    for addr,reader in readers.items():
        print("Stopping",addr,end="....")
        reader.stop()
        print("....done")


def read_log(fname):
    return pd.read_hdf(fname,key="/data")

def read_logs(fnames):
    data = read_log(fnames[0])
    for f in fnames[1:]:
        data = data.append(read_log(f))
    return data

def read_folder(folder):
    files = pathlib.Path(folder).glob("*.h5")
    files = list(files)
    files.sort()
    return read_logs(list(files))



def test_generation(port=1234,every=10):
    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    sock.bind('tcp://*:%d'%port)
    try:
        t0 = time.time()
        n = 0
        while True:
            t1 = time.time()
            data = { "time" : datetime.datetime.now(),
             "sensor1/value1" : random.random(),
             "sensor1/value2" : random.random(),
             "sensor2/value1" : random.random(),
             "sensor2/value2" : random.random(),
             }
            sock.send_pyobj(data)
            print(data)
            tneed = time.time()-t1
            n += 1
            if tneed<every: time.sleep(every-tneed)
    except KeyboardInterrupt:
        sock.close()
        dt = time.time()-t0
        print("Made data at %s Hz"%(n/dt))



