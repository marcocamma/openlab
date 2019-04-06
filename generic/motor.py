""" Module to abstract motor """
from __future__ import print_function,division


import numpy as np
import os
import logging
import time
import sys

from ..utils import  yaml_storage
from ..utils import  keypress
from ..utils.strings import notice

STORAGE = None


logging.basicConfig(format='%(asctime)s %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

__author__ = "Marco Cammarata"
__copyright__ = "Copyright 2016, Marco Cammarata"
__version__ = "1.0"
__maintainer__ = "Marco Cammarata"
__email__ = "marcocamma@gmail.com"


class Motor:
  
    def __init__(self,mne,mvd,wmd,set=None,mvr=None,wait=None,precision=1e-3,
            parents=None,polltime=0.01,timeout=1,units="mm"):
        self.mne = mne
        self._mvd = mvd
        self.wmd = wmd
        #self.setd = setd
        self.precision = precision
        self.parents = parents
        self.units=units

        # use default macros for wait,mvr,set if not passed
        self.mvr = mvr if mvr is not None else self._mvr
        self.wait = wait if wait is not None else self._wait
        self.set = set if set is not None else self._set

        self.mv = self.move
        self.dial_target_pos = None
        self.polltime = polltime
        self.timeout = timeout
        if STORAGE is None: # could (and should) be set at setup level
            globals()['STORAGE'] = yaml_storage.Storage(filename="./offsets/", autosave=True)

    def mvd(self,value):
        """ this function should offset work to a thread ... to be compatible with blocking mvd """
        self._mvd(value)

    def _set(self,value):
        current_dial = self.wmd()
        newoffset = current_dial-value
        STORAGE['%s/offset'%self.mne] = newoffset

    def _mvr(self,how_much,wait=False):
        pos = self.wm()
        self.move(pos+how_much,wait=wait)

    def _get_offset(self):
        offset = STORAGE['%s/offset'%self.mne]
        return offset

    def wm(self,as_dial=False,as_str=False):
        value = self.wmd()
        if not as_dial:
            offset = STORAGE['%s/offset'%self.mne]
            value = value - offset
        if as_str:
            value = str(np.round( value, int(-np.log10(self.precision)+1 )))
        return value

    def move(self,value,wait=False):
        offset = STORAGE['%s/offset'%self.mne]
        dial = value + offset
        self.dial_target_pos = dial
        self.mvd(dial)
        if wait: self.wait()

    def _wait(self,timeout=None):
        if self.dial_target_pos is None: return
        if timeout is None: timeout = self.timeout
        t0 = time.time()
        while( time.time()-t0 < timeout and np.abs(self.wmd()-self.dial_target_pos)>self.precision) :
            time.sleep(self.polltime)
        if np.abs(self.wmd()-self.dial_target_pos)>self.precision:
            logger.warn("Motor, %s, did not reached target position"%self.mne)

    def __repr__(self):
        pos = str(np.round( self.wm(), int(-np.log10(self.precision)+1 )))
        posd = str(np.round( self.wmd(), int(-np.log10(self.precision)+1 )))
        #s = f"motor {self.mne}, position {pos}, dial {posd}"
        u = self.units
        s = "motor %s, position %s %s, dial %s %s" %(self.mne,pos,u,posd,u)
        return s

    def tweak(self,initial_step=None):
        if initial_step is None: initial_step = self.precision*5
        step = initial_step
        help = "q = exit; up = step*2; down = step/2, left = neg dir, right = pos dir\n"
        help = help + "g = go abs, s = set"
        print("tweaking motor %s" % self.get_info_str())
        oldstep = 0
        k=keypress.KeyPress()
        while (k.isq() is False):
            if (oldstep != step):
                nstr = "stepsize: %f" % step
                notice(nstr)
                oldstep = step
            k.waitkey()
            if   k.isu(): step = step*2.
            elif k.isd(): step = step/2.
            elif k.isr():
                self.mvr(step,wait=True)
            elif k.isl():
                self.mvr(-step,wait=True)
            elif ( k.iskey("g") ):
                print("enter absolute position (char to abort go to)")
                sys.stdout.flush()
                v=sys.stdin.readline()
                try:
                    v = float(v.strip())
                    self.move(v)
                except:
                    print("value cannot be converted to float, going back ...")
                    sys.stdout.flush()
            elif k.iskey("s"):
                print("enter new set value (char to abort setting)")
                sys.stdout.flush()
                v=sys.stdin.readline()
                try:
                    v = float(v[0:-1])
                    self.set(v)
                except:
                    print("value cannot be converted to float, going back ...")
                    sys.stdout.flush()
            elif k.isq():
                break
            print("Current position",self.wm(as_str=True))
        else:
            print(help)
        print("final position: %s" % self.get_info_str())


    def get_info_str(self,prepend=""):
        ret = ""
        offset = STORAGE['%s/offset'%self.mne]
        ret += "%smotor name: %s\n"%(prepend,self.mne)
        ret += "%soffset: %s %s\n"%(prepend,str(offset),self.units)
        ret += "%sprecision: %s %s\n"%(prepend,str(self.precision),self.units)
        if self.parents is not None:
            ret += "%sParent motor(s):\n"%prepend
            parents = self.parents
            if isinstance(parents,Motor): parents = (parents,)
            for parent in parents:
                ret += "  **********\n"
                ret += parent.get_info_str(prepend="  ")
        return ret

        
    def __call__(self,value):
        self.move(value)


def get_fakemot():
    class FakeMotor():
        def __init__(self,verbose=False):
            self.pos=0
            self.verbose=verbose
        def mv(self,pos):
            if self.verbose:
                print("this is the dial of the fake motor, I am asked to move to",pos)
            self.pos = pos
        def wm(self):
            return self.pos
    fake = FakeMotor()
    m = Motor("fakemot",fake.mv,fake.wm)
    return m

if __name__ == "__main__":
    get_fakemot()
