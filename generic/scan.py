import pathlib
from collections.abc import Iterable
import itertools
import numpy as np
import datetime
import tqdm
import datastorage
from datastorage import DataStorage as ds


def _get_npoints(positions):
    positions = np.asarray(positions)
    if positions.ndim == 1: positions = positions[:,np.newaxis]
    npoints = np.asarray([len(np.unique(axis)) for axis in positions.T])
    if len(npoints) == 1: npoints = int(npoints)
    return npoints


def now(as_str=True):
    n = datetime.datetime.now()
    if as_str: n = str(n)
    return n

def _general_scan(motors=None,positions=None,acquire=None,fname=None,force=False):
    """
    general porpouse scanning macro

    Parameters
    ----------
    motors : list|tuple
        each item should be a motor/stage
    positions : 2D arrayable
        2D array scan_point,num_mot
    acquire : function
        a function that must be provided, it must return a dictionary with
        items to pack as output
        the keys starting with _ will only be included one (and not in all
        scan points)
        example: def read(): time.sleep(1); return dict(val=3,_x=np.arange(10))
        like data structure
    """

    if fname is None:
        print("Must give fname")
        return

    # motors must be a list
    if not isinstance(motors,Iterable): motors = [motors,]

    # positions bust be 2D-like array
    positions = np.asarray( positions )
    if positions.ndim == 1: positions = positions[:,np.newaxis]

    info = ds()
    info.num_motors = len(motors)
    info.motors = [m.mne for m in motors]
    info.motors_paramters = ds()
    for m in motors:
        info.motors_paramters[m.mne] = m.get_info_str()

    info.positions = np.squeeze(positions)
    info.npoints_per_axis = _get_npoints(positions)
    info.time_start = now()


    # ensure we are using pathlib
    fname = pathlib.Path(fname)

    # check if file exists
    if fname.exists() and not force:
        print("File %s exists, returning"%fname)
        return
    else:
        fname.parent.mkdir(exist_ok=True,parents=True)
        print("Will save in",str(fname))

    data_buffer = []
    # can't use enumerate because of tqdm missing support (at least for 4.11)
    for iscan in tqdm.trange(len(positions)):
        acquire_positions = positions[iscan]

        tosave = ds(info=info)
        # first axis is scan points
        tosave.positions = np.squeeze(positions[:iscan+1])
        tosave.npoints_per_axis = _get_npoints(positions[:iscan+1])

        # ask to move motors to positions
        for motor,position in zip(motors,acquire_positions):
            motor.move(position)

        # wait until they arrive
        for motor in motors: motor.wait()

        _data = acquire()
        data_buffer.append( _data )

        for key in _data:
            if key[0] == "_": continue
            tosave[key] = np.asarray( [data[key] for data in data_buffer] )
            tosave[key] = np.squeeze( tosave[key] )
        keys_tosave_once = list(_data.keys())
        keys_tosave_once = [key for key in keys_tosave_once if key[0] == "_"]

        for key in keys_tosave_once:
            tosave[key.strip("_")] = _data[key]
        tosave.info.time_last_save = now()

        tosave.save(str(fname))
    return tosave



def ascan(motor,start,stop,N,**pars):
    """ absolute 1D scan """
    positions = np.linspace(start,stop,N+1)
    return _general_scan(motors=motor,positions=positions,**pars)

def rscan(motor,start,stop,N,**pars):
    """ relative 1D scan """
    positions = np.linspace(start,stop,N+1)+motor.wm()
    return _general_scan(motors=motor,positions=positions,**pars)

def a2scan(m1,s1,e1,n1,m2,s2,e2,n2,**pars):
    motors = [m1,m2]
    p1 = np.linspace(s1,e1,n1+1)
    p2 = np.linspace(s2,e2,n2+1)
    positions = np.asarray( list(itertools.product(p1,p2)) )
    return _general_scan(motors=motors,positions=positions,**pars)

def r2scan(m1,s1,e1,n1,m2,s2,e2,n2,**pars):
    motors = [m1,m2]
    p1 = np.linspace(s1,e1,n1+1)+m1.wm()
    p2 = np.linspace(s2,e2,n2+1)+m2.wm()
    positions = np.asarray( list(itertools.product(p1,p2)) )
    return _general_scan(motors=motors,positions=positions,**pars)


def read_2dscan(fname,keys="all"):
    data = datastorage.read(fname)
    if data.info.num_motors != 2:
        print("Did not find 2 motors names in num_motors")
        return
    size = list(_get_npoints(data.info.positions))
    for iaxis,axis in enumerate(data.info.motors):
        data[axis] = data.info.positions[:,iaxis].reshape(size)
    if keys == "all" : keys = data.keys()
    for key in keys:
        if isinstance(data[key],np.ndarray):
            try:
                newsize = size.copy()
                if data[key].ndim == 2: newsize.append( -1 )
                data[key] = data[key].reshape( newsize )
            except ValueError:
                pass
    return data



