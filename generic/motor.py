""" Module to abstract motor """
from __future__ import print_function,division


import numpy as np
import os
import logging
import time

from ..utils import  yaml_storage

STORAGE = None


logging.basicConfig(format='%(asctime)s %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

__author__ = "Marco Cammarata"
__copyright__ = "Copyright 2016, Marco Cammarata"
__version__ = "1.0"
__maintainer__ = "Marco Cammarata"
__email__ = "marcocamma@gmail.com"


class Motor:
  
    def __init__(self,mne,mvd,wmd,mvr=None,wait=None,precision=1e-3,polltime=0.01,timeout=1):
        self.mne = mne
        self._mvd = mvd
        self.wmd = wmd
        #self.setd = setd
        self.precision = precision
        if wait is None:
            self.wait = self._wait; # defined in this macro
        else:
            self.wait = wait
        self.mvr = mvr if mvr is not None else self._mvr
        self.mv = self.move
        self.dial_target_pos = None
        self.polltime = polltime
        self.timeout = timeout
        if STORAGE is None: # could (and should) be set at setup level
            globals()['STORAGE'] = yaml_storage.Storage(filename="./offsets/", autosave=True)

    def set(self,value):
        current_dial = self.wmd()
        newoffset = current_dial-value
        STORAGE['%s/offset'%self.mne] = newoffset

    def mvd(self,value):
        """ this function should offset work to a thread ... to be compatible with blocking mvd """
        self._mvd(value)

    def _mvr(self,how_much):
        pos = self.wm()
        self.move(pos+how_much)


    def wm(self,as_dial=False):
        value = self.wmd()
        if not as_dial:
            offset = STORAGE['%s/offset'%self.mne]
            value = value - offset
        return value

    def move(self,value,wait=False):
        offset = STORAGE['%s/offset'%self.mne]
        dial = value + offset
        self.dial_target_pos = dial
        self.mvd(dial)

    def _wait(self,timeout=None):
        if self.dial_target_pos is None: return
        if timeout is None: timeout = self.timeout
        t0 = time.time()
        while( time.time()-t0 < timeout and np.abs(self.wmd()-self.dial_target_pos)>self.precision) :
            time.sleep(self.polltime)
        if np.abs(self.wmd()-self.dial_target_pos)>self.precision:
            logger.warn("Motor, %s, did not reached target position"%self.mne)

    def __repr__(self):
        pos = str(np.round( self.wm(), int(-np.log10(self.precision) )))
        posd = str(np.round( self.wmd(), int(-np.log10(self.precision) )))
        #s = f"motor {self.mne}, position {pos}, dial {posd}"
        s = "motor %s, position %s, dial %s" %(self.mne,pos,posd)
        return s

    def __call__(self,value):
        self.move(value)
