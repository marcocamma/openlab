#from openlab.oscilloscopes.lecroy import lecroy
try:
    #from ...generic import scan
    from openlab.generic import scan
except ImportError:
    import sys
    sys.path.insert(0,"/users/opid09/users/marco/openlab/")
    from openlab.generic import scan
#    from ...generic import scan

try:
    from . import config
except ImportError:
    import config

from openlab.utils import archiver

from openlab.oscilloscopes import tektronix
import time
from openlab.generic import motor
motor.STORAGE = config.storage
STORAGE=config.storage
from datastorage import DataStorage
import numpy as np
import machine_bpm
CH=12.3984192976


CH_PD1 = 1
CH_PD3 = 4

scope = tektronix.TektronixScope(("id09msoscope",4000))
#scope.prepare_fastframe(10)

from bliss import config
config.get_sessions_list()
config = config.static.get_config()
bliss_mono1= config.get("mono1")
bliss_ss2d= config.get("ss2d")
bliss_mono1e= config.get("mono1e")
bliss_mono1t= config.get("mono1t")
bliss_machinfo = config.get("machinfo")


def bliss_to_openlab(bliss_motor,mne=None,precision=1e-6,units="mm",parents=None):
    def _wm():
        return bliss_motor.position

    def _ready():
        return bliss_motor == False

    if mne is None: mne = bliss_motor.name

    def move_with_retry(value):
        for i in range(10):
            try:
                bliss_motor.move(value)
                break
            except Exception as e:
                print("Failed to move (attempt %d)"%i,mne,"error was:",str(e))



    mot = motor.Motor(mne,
        move_with_retry,
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

phg = bliss_to_openlab(config.get("phg"))
pvg = bliss_to_openlab(config.get("pvg"))
mono1t = bliss_to_openlab(config.get("mono1t"))

mono1 = bliss_to_openlab(bliss_mono1,"mono1",
        precision=1e-6,units="deg")


_d_111 = 3.13465698029
_d_333 = _d_111/3

WAIT_BEFORE_ACQUIRE=1

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

def get_sr():
    try:
        srcur = bliss_machinfo.counters["current"].value
    except:
        srcur = bliss_machinfo.counters["current"].value
    return srcur


def acquire_traces(use_single=True):
    time.sleep(WAIT_BEFORE_ACQUIRE)
    
    dt=float(scope.ask("HOR?")["HORIZONTAL:ACQDURATION"])

    scope.acquire_single_trigger()
    data_bpm = machine_bpm.acquire(
            ["XPosBuffer","ZPosBuffer","XAngleBuffer","ZAngleBuffer"],
            dt=dt,
            concatenate=False
            )

    scope.wait_for_end() # somehow not working
    #time.sleep(2)
    y1 = scope.get_ch(CH_PD1)
    y2 = scope.get_ch(CH_PD3)
    x = scope.get_xaxis()
    data = dict(
            _xdata=x,
            y1=y1,
            y2=y2,
            mono=bliss_mono1.position,
            mymono=mono1.wm(),
            phg=phg.wm(),
            pvg=pvg.wm(),
            srcur = get_sr(),
            datetime=str(archiver.now())
            )
    data.update(data_bpm)
    return data




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

def ascan(motor,start=0.05,stop=-0.05,n=25):
    # MONO_TO_SCAN.move(19.9659) # for 333
    last_run = STORAGE["last_run"]
    new_run = last_run+1
    d=scan.ascan(motor,start,stop,n,acquire=acquire_traces,
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

def mono_ascan(start=19.98,stop=20.034,n=25):
    last_run = STORAGE["last_run"]
    new_run = last_run+1
    d=scan.ascan(MONO_TO_SCAN,start,stop,n,acquire=acquire_traces,
            fname="data/run%03d.h5"%new_run)
    STORAGE["last_run"]=new_run

def mono_step(dangle_urad=100,collect_before=20):
    # always approach from same side (smaller energy)
    #mono1.move(start+0.2)
    #mono1.wait()
    #mono1.move(start)
    #time.sleep(100)
    #input("prepare scope (10s window, single); ok to start")
    start=mono1.wm()
    scope.acquire_single_trigger()
    time.sleep(collect_before)
    mono1.mvr(np.rad2deg(dangle_urad*1e-6))
    scope.wait_for_end()
    ch1 = scope.get_ch(CH_PD1)
    ch2 = scope.get_ch(CH_PD3)
    x = scope.get_xaxis()
    last_run = STORAGE["last_run"]
    new_run = last_run+1
    fname = fname="data/run%03d.h5"%new_run
    d = DataStorage(start=start,dangle_urad=dangle_urad,xdata=x,y1=ch1,y2=ch2)
    d.save(fname)
    STORAGE["last_run"]=new_run
    print("saved",fname)


    pass
