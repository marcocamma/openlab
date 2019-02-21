#from openlab.oscilloscopes.lecroy import lecroy
from . import config
import openlab.stages
from openlab.generic import motor
motor.STORAGE = config.storage
from openlab.generic import delay_stage as _delay_stage
from openlab.oscilloscopes.lecroy import LeCroyScope
from datastorage import DataStorage


try:
    scope = LeCroyScope("129.20.84.103")
except OSError:
    scope = None


esp301 = openlab.stages.newportESP.ESP("/dev/ttyUSB0")
if esp301 is not None:
    xyz = DataStorage()
    for axis in (1,2,3):
        esp301_ax = esp301.axis(axis)
        name = "axis%d"%axis
        esp301_ax.on()
        stage_mot = motor.Motor(name,
            esp301_ax.move_to,
            esp301_ax.get_position,
            wait= esp301_ax.wait
        )
        xyz[name] = stage_mot
else:
    xyz = None

stage = openlab.stages.ZaberStage("/dev/ttyACM0")
if stage is not None:
    stage_mot = motor.Motor("stage",
        stage.move,
        stage.get_position,
        wait= stage.wait,
        precision=1e-4, # precision is used for printing only
    )
    delay_stage = _delay_stage.DelayStage(stage_mot)
else:
    delay_stage = None
