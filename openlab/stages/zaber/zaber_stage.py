# https://www.zaber.com/wiki/Manuals/X-LRQ-DEC
# https://www.zaber.com/wiki/Manuals/ASCII_Protocol_Manual
import sys
import types
import time
import zaber.serial
import os

# read doc
doc_file = os.path.join( os.path.dirname(__file__), "zaber_serial_doc.txt" )

with open(doc_file,"r") as f:
    help_rows = f.readlines()

# remove empty lines
help_rows = [row for row in help_rows if len(row)>0]
# remove comments
help_rows = [row for row in help_rows if row.lstrip()[0] != "#"]

help_rows = [ [elem.strip() for elem in row.split("\t")] for row in help_rows]

MAXSPEED = 153600

class Cmd:
    __slots__ = "setting","scope","writable","doc","as_type"
    def __init__(self,setting="pos", scope="Axis", writable=False, doc="", as_type=int):
        self.setting=setting
        self.scope=scope
        self.writable=writable
        self.as_type=as_type
        self.doc=doc



CMD_DICT = dict()
for help_row in help_rows:
    if len(help_row) == 5:
        setting,scope,writable,_,description=help_row
        as_type = None
    else:
        setting,scope,writable,_,description,as_type=help_row
        if as_type.lower() == "int":
            as_type = int
    writable = writable == "Yes"
    temp = Cmd(setting=setting, scope=scope, writable=writable, 
            doc=description, as_type=as_type)
    name = setting.replace(".","_")
    CMD_DICT[name]=temp



def _prepare_ans(ans,full_answer=False,as_type=None):
    if full_answer:
        return ans
    else:
        ans = ans.data
        if as_type is not None:
            ans = as_type(ans)
        return ans

def make_property(name,doc=None):
    return property(
        fget = lambda self: self._get(name),
        fset = lambda self, value: self._set(name,value),
        doc  = CMD_DICT[name].doc
    )


class ZaberStageProperties:
    for cmd in CMD_DICT:
        locals()[cmd] = make_property(cmd)

    def __init__(self,stage):
        self.stage = stage
        self._get = stage._get
        self._set = stage._set


class ZaberStage:

    def __init__(self,port="/dev/ttyACM1",axis=1,screw_pitch=1.270):
        """
        Parameters
        ----------
        screw_pitch: in mm
        """
        try:
            port = zaber.serial.AsciiSerial(port)
            device = zaber.serial.AsciiDevice(port,1)
        except Exception as err:
            print("Could not connect to zaber stage, error was: ",err)
            return None
        self.device = device
        self.axis = device.axis(axis)
        self.settings = ZaberStageProperties(self)
        #self.commands = datastorage.DataStorage()
        self.fullstep_resolution = screw_pitch/self._get("cloop_steps")
        self.microstep_resolution = self.fullstep_resolution/self._get("resolution")

        self.encoder_resolution = screw_pitch/self._get("cloop_counts")

    def help(self):
        commands = list(CMD_DICT.keys())
        commands.sort()
        for cmd in commands:
            print("%20s %s"%(cmd,CMD_DICT[cmd].doc))

    def _get(self,cmd,full_answer=False,as_type="try"):
        if cmd not in CMD_DICT:
            raise ValueError(cmd + " does not exist")
        else:
            cmd = CMD_DICT[cmd]
        if cmd.scope == "Axis":
            ans=self.axis.send("get %s"%cmd.setting)
        else:
            ans=self.device.send("get %s"%cmd.setting)
        ans = _prepare_ans(ans,full_answer=full_answer,as_type=cmd.as_type)
        if as_type == "try":
            try:
                ans = float(ans)
            except ValueError:
                pass
        return ans



    def _set(self,cmd,value,full_answer=False):
        if cmd not in CMD_DICT:
            raise ValueError(cmd + " does not exist")
        else:
            cmd = CMD_DICT[cmd]
        if cmd.as_type is not None:
            value = cmd.as_type(value)
        if cmd.scope == "Axis":
            ans=self.axis.send("set %s %s"%(cmd.setting,value))
        else:
            ans=self.device.send("set %s %s"%(cmd.setting,value))
        return _prepare_ans(ans,full_answer=full_answer,as_type=cmd.as_type)

    def home(self):
        self.axis.home()

    def get_position(self):
        return self._get("pos")*self.microstep_resolution

    def move(self,value,wait=False):
        steps = value/self.microstep_resolution
        self.axis.send("move abs %d"%int(steps))
        if wait: self.wait()

    def moverel(self,distance,wait=False):
        steps = distance/self.microstep_resolution
        self.axis.send("move rel %d"%int(steps))
        if wait: self.wait()

    def moveatspeed(self,speed,wait=False):
        steps_per_sec = speed/self.microstep_resolution
        self.axis.send("move vel %d"%int(steps_per_sec))
        if wait: self.wait()

    def wait(self):
        done = False
        while not done:
            done = self.axis.send('').device_status == "IDLE"
            time.sleep(0.01)

    def stop(self):
        self.axis.send("stop")

    def setdefaultspeed(self):
        self.maxspeed = MAXSPEED

    def enc_pos(self):
        return self._get("encoder_count_calibrated")*self.encoder_resolution

    def as_openlab_motor(self,name="motor",precision_printing=1e-4):
        from openlab.generic import motor
        stage = motor.Motor(name,
                self.move,
                self.get_position,
                wait= self.wait,
                precision=precision_printing,
                parents=self
            )
        return stage


