""" Module to communicate with Thorlabs MC2000 optical chopper controller; 
 The interface is (and probably will) be very minimal; consult manual at:
 https://www.thorlabs.de/thorcat/18400/MC2000-Manual.pdf for help especially
 for mapping of blades """
from __future__ import print_function,division


import serial
import time
import os
import logging

logging.basicConfig(format='%(asctime)s %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

__author__ = "Marco Cammarata"
__copyright__ = "Copyright 2016, Marco Cammarata"
__version__ = "1.0"
__maintainer__ = "Marco Cammarata"
__email__ = "marcocamma@gmail.com"


_serialSettings = dict( baudrate = 115200, bytesize = 8, parity = "N", stopbits = 1, timeout=0.5 )

_remote_commands = dict(
  freq  = "Internal reference frequency",
  phase = "Phase adjust (0-360 deg)",
  blade = "Blade type (n is number indicated in section 6.4)",
  nharmonic = "Harmonic Multiplier applied to external reference frequency",
  dharmonic = "Harmonic Divider applied to external reference frequency",
  ref       = "Reference mode (0=internal, 1=external)",
  output    = "Output reference mode (0=target, 1=actual)"
)


def make_property(name,doc=None):
    return property(
        fget = lambda self: self._get_value(name),
        fset = lambda self, value: self._set_value(name,value),
        doc  = doc
    )

class MC2000(object):
  for cmd,doc in _remote_commands.items():
    locals()[cmd] = make_property(cmd,doc)

  
  def __init__(self,port="/dev/ttyUSB0",sleepTime=0.02,verbose=False):
    """ port is usually /dev/ttyUSB0, /dev/ttyUSB1, /dev/ttyUSB? in linux"""
    self._port = port
    self._sleepTime = sleepTime
    self.connect()
    if verbose: logger.setLevel( logging.INFO & logging.DEBUG)

  def help(self):
    s = []
    for cmd,doc in _remote_commands.items():
      s.append( "%10s : %s" % (cmd,doc) )
    print(__doc__)
    print("Commands are:")
    print("\n".join(s))
    

  def connect(self):
    logger.info("Trying to connect to port %s"%self._port)
    if os.path.exists(self._port):
      self._s = serial.Serial(port=self._port,**_serialSettings)
      self.id = self._get_value("id",asInt=False)
      logger.info("Connection OK")
    else:
#      print("The port %s does not exist; connect device and use .connect method" % self._port)
      logger.warn("Connection Failed; the port %s does not exist; connect device and use .connect method" % self._port)
      self._s = None
      self.id = None

  def _send(self,cmd):
    """ Send cmd taking care of carriage return and encoding """
    string = cmd+"\r"
    logger.debug("Sending '%s'"%string)
    self._s.write( (cmd+"\r").encode() )

  def _read(self):
    """ read answer """
    ret = self._s.read_all()
    logger.debug("Read '%s'"%ret)
    return ret

  def _get_value(self,cmd,asInt=True):
    """ Query command (like 'status','freq', etc.); cast reply as integer by default """
    query = cmd + "?"
    self._send(query)
    time.sleep(self._sleepTime)
    ret = self._read()
    # answer is cmd?\rVALUE\r
#    ans = ret[len(query)+1:-3]
    ans = ret.split(b'\r')[-2]
    if asInt:
      ans=int(ans)
      logger.debug("Interpreting answer as %d"%ans)
    else:
      logger.debug("Interpreting answer as '%s'"%ans)
    return ans

  def _set_value(self,cmd,value):
    """ set a given variable to value (by formatting the string as variable=value """
    v = cmd + "=" + str(value)
    logger.debug("Set command '%s'"%v)
    self._send(v)
    time.sleep(self._sleepTime)
    ret = self._read()
    return None

  def isRunning(self): return self._get_value("enable") == 1
  def start(self): self._set_value("enable",1)
  def stop(self):  self._set_value("enable",0)
