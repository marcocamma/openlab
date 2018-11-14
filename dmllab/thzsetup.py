#from ..oscilloscopes.lecroy import lecroy
from ..stages.newport import newportESP
from ..generic import motor
from ..utils import yaml_storage
from ..generic import delay_stage as _delay_stage
from ..oscilloscopes.lecroy import LeCroyScope
import config

motor.STORAGE = config.offset_storage


try:
    scope = LeCroyScope("129.20.76.26")
except OSError:
    scope = None


esp301 = newportESP.ESP("/dev/ttyUSB0")
if esp301 is not None:
    esp301_ax1 = esp301.axis(1)
    esp301_ax1.on()

    stage_mot = motor.Motor("stage1",
        esp301_ax1.move_to,
        esp301_ax1.get_position,
        wait= esp301_ax1.wait
    )

    delay_stage = _delay_stage.DelayStage(stage_mot)
else:
    delay_stage = None
