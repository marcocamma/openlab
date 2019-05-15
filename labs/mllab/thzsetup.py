#from openlab.oscilloscopes.lecroy import lecroy
from . import config
import openlab.stages
from openlab.generic import motor
motor.STORAGE = config.storage
from openlab.generic import delay_stage as _delay_stage
from openlab.oscilloscopes.lecroy import LeCroyScope
from datastorage import DataStorage

print("Laboratory convention right handed,y=vertical,z=downbeam")

try:
    scope = LeCroyScope("129.20.84.103")
except OSError:
    scope = None


esp301 = openlab.stages.newportESP.ESP("/dev/ttyUSB0")
if esp301 is not None:
    xyz = DataStorage()
    names = "y x z".split()
    for (axis,name) in zip((1,2,3),names):
        xyz[name] = esp301.axis(axis).as_openlab_motor(name=name)
else:
    xyz = None

stage = openlab.stages.ZaberStage("/dev/ttyACM0")
if stage is not None:
    stage_mot = stage.as_openlab_motor(name="stage",precision_printing=1e-4)
    delay_stage = _delay_stage.delaystage(stage_mot,precision=2e-3)
else:
    delay_stage = None
