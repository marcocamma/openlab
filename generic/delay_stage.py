""" Module to abstract delay stage """
from __future__ import print_function,division


import numpy as np
import os
import logging


logging.basicConfig(format='%(asctime)s %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

__author__ = "Marco Cammarata"
__copyright__ = "Copyright 2016, Marco Cammarata"
__version__ = "1.0"
__maintainer__ = "Marco Cammarata"
__email__ = "marcocamma@gmail.com"


from .motor import Motor,get_fakemot

from scipy.constants import speed_of_light as c_light

def _mm_to_ps(pos_mm,bounces=1):
    pos_m = pos_mm*1e-3
    delay_s = pos_m*2*bounces/c_light
    delay_ps = delay_s*1e+12
    return delay_ps

def _ps_to_mm(delay_ps,bounces=1):
    delay_s = delay_ps*1e-12
    pos_m = delay_s/2/bounces*c_light
    pos_mm = pos_m*1e3
    return pos_mm

class DelayStage:
  
    def __init__(self,motor,bounces=1):
        self._motor = motor
        self.bounces = bounces
        #self.mne = f"delay_{self._motor.mne}"
        self.mne = "delay_%s"%self._motor.mne


    def move(self,delay_ps,wait=False):
        pos = _ps_to_mm(delay_ps,bounces=self.bounces)
        self._motor.move(pos)
        if wait: self.wait()

    def mvr(self,delta_delay_ps):
        self._motor.mvr(_ps_to_mm(delta_delay_ps))

    def wm(self):
        stage_mm = self._motor.wm()
        delay_ps = _mm_to_ps(stage_mm,bounces=self.bounces)
        return delay_ps

    def wmd(self):
        stage_mm = self._motor.wmd()
        delay_ps = _mm_to_ps(stage_mm)
        return delay_ps

    def wait(self):
        self._motor.wait()

    def get_info_str(self):
        return self._motor.get_info_str()

    def set(self,value):
        pos = _ps_to_mm(value)
        self._motor.set(pos)

    def __repr__(self):
        pos = str(np.round( self.wm(), 3))
        posd = str(np.round( self.wmd(), 3))
        #s = f"{self.mne}, position {pos} ps, dial {posd} ps"
        s = "%s, position %s ps, dial %s ps"%(self.mne,pos,posd)
        return s

    def __call__(self,value):
        self.move(value)


def delaystage(motor,bounces=1):
    def wmd():
        stage_mm = motor.wm()
        delay_ps = _mm_to_ps(stage_mm,bounces=bounces)
        return delay_ps

    def mvd(delay_ps,wait=False):
        pos = _ps_to_mm(delay_ps,bounces=bounces)
        motor.move(pos,wait=wait)

    def set(delay_ps):
        """ set current position to delay_ps """
        pos = _ps_to_mm(delay_ps,bounces=bounces)
        motor.set(pos)

    mne = "delay_%s"%motor.mne
    precision = _mm_to_ps(motor.precision)
    m = Motor(mne,mvd,wmd,set=set,precision=precision,parents=motor,units="ps")
    return m


def get_fakestage():
    m = get_fakemot()
    s = delaystage(m)
    return s
