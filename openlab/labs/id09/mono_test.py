#from openlab.oscilloscopes.lecroy import lecroy
from ...generic import scan
from . import config
import tektronix
import time
from openlab.generic import motor
motor.STORAGE = config.storage
STORAGE=config.storage
from datastorage import DataStorage
import numpy as np
CH=12.3984192976


scope = tektronix.TektronixScope(("id09msoscope",4000))
#scope.prepare_fastframe(10)

from bliss import config
config.get_sessions_list()
config = config.static.get_config()
bliss_mono1= config.get("mono1")
bliss_ss2d= config.get("ss2d")
bliss_mono1e= config.get("mono1e")


def bliss_to_openlab(bliss_motor,mne=None,precision=1e-6,units="mm",parents=None):
    def _wm():
        return bliss_motor.position

    def _ready():
        return bliss_motor == False

    if mne is None: mne = bliss_motor.name

    mot = motor.Motor(mne,
        bliss_motor.move,
        _wm,
        set=None,
        mvr=None,
        wait=_ready,
        precision=precision,
        parents=None,
        polltime=0.01,
        timeout=1,
        units=units
        )
    return mot


ss2l = bliss_to_openlab(config.get("ss2l"))
ss2d = bliss_to_openlab(config.get("ss2d"))


mono1 = bliss_to_openlab(bliss_mono1,"mono1",
        precision=1e-6,units="deg")


_d_111 = 3.13465698029
_d_333 = _d_111/3

WAIT_BEFORE_ACQUIRE=0.1

def _mv_333(energy):
    a = CH/(2*_d_333*energy)
    a = np.arcsin(a)
    a = np.rad2deg(a)
    mono1.mv(a)

def _wm_333():
    angle = mono1.wmd()
    angle = np.deg2rad(angle)
    energy = CH/(2*_d_333*np.sin(angle))
    return energy

def _mv_111(energy):
    a = CH/(2*_d_111*energy)
    a = np.arcsin(a)
    a = np.rad2deg(a)
    mono1.mv(a)

def _wm_111():
    angle = mono1.wmd()
    angle = np.deg2rad(angle)
    energy = CH/(2*_d_111*np.sin(angle))
    return energy


def acquire_traces(nshots=100,use_single=True):
    time.sleep(WAIT_BEFORE_ACQUIRE)
    scope.acquire_single_trigger()
    scope.wait_for_end()
    y1 = scope.get_ch(1)
    y2 = scope.get_ch(2)
    x = scope.get_xaxis()
    return dict( _xdata=x,y1=y1,y2=y2,mono=bliss_mono1.position,mymono=mono1.wm())


mono333 = motor.Motor("mono333",
        _mv_333,
        _wm_333,
        set=None,
        mvr=None,
        wait=mono1.wait,
        precision=1e-5,
        parents=[mono1,],
        polltime=0.01,
        timeout=1,
        units="keV")

mono111 = motor.Motor("mono111",
        _mv_111,
        _wm_111,
        set=None,
        mvr=None,
        wait=mono1.wait,
        precision=1e-5,
        parents=[mono1,],
        polltime=0.01,
        timeout=1,
        units="keV")


MONO_TO_SCAN=mono111

def rscan(motor,start=0.05,stop=-0.05,n=25):
    # MONO_TO_SCAN.move(19.9659) # for 333
    last_run = STORAGE["last_run"]
    new_run = last_run+1
    d=scan.rscan(motor,start,stop,n,acquire=acquire_traces,
            fname="data/run%03d.h5"%new_run)
    STORAGE["last_run"]=new_run



def mono_rscan(start=0.05,stop=-0.05,n=25):
    # MONO_TO_SCAN.move(19.9659) # for 333
    MONO_TO_SCAN.move(20.025) # for 111
    last_run = STORAGE["last_run"]
    new_run = last_run+1
    d=scan.rscan(MONO_TO_SCAN,start,stop,n,acquire=acquire_traces,
            fname="data/run%03d.h5"%new_run)
    STORAGE["last_run"]=new_run

def mono_ascan(start=19.978+0.05,stop=19.978-0.05,n=25):
    last_run = STORAGE["last_run"]
    new_run = last_run+1
    d=scan.ascan(MONO_TO_SCAN,start,stop,n,acquire=acquire_traces,
            fname="data/run%03d.h5"%new_run)
    STORAGE["last_run"]=new_run

def mono_step(start=19.978,de=1e-3):
    MONO_TO_SCAN.move(start)
    time.sleep(5)
    input("prepare scope (10s window, single); ok to start")
    scope.acquire_single_trigger()
    time.sleep(1)
    MONO_TO_SCAN.mvr(de)
    scope.wait_for_end()
    ch1 = scope.get_ch(1)
    ch2 = scope.get_ch(2)
    x = scope.get_xaxis()
    last_run = STORAGE["last_run"]
    new_run = last_run+1
    fname = fname="data/run%03d.h5"%new_run
    d = DataStorage(start=start,denergy=de,xdata=x,y1=ch1,y2=ch2)
    d.save(fname)
    STORAGE["last_run"]=new_run


    pass
