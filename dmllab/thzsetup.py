from ..lecroy import lecroy
from ..newport import newportESP
from ..generic import motor
from ..utils import yaml_storage
from ..generic import delay_stage as _delay_stage
from ..oscilloscopes.lecroy import LeCroyScope

import pathlib
p = pathlib.Path(__file__).parent / "offsets"

motor.STORAGE = yaml_storage.Storage(filename=str(p),autosave=True)

try:
    scope = lecroy.thz_lecroy()
except OSError:
    scope = None


esp301 = newportESP.ESP("/dev/ttyUSB0")

esp301_ax1 = esp301.axis(1)
esp301_ax1.on()

stage_mot = motor.Motor("stage1",
        esp301_ax1.move_to,
        esp301_ax1.get_position,
        wait= esp301_ax1.wait
        )



thzscope = LeCroyScope("129.20.76.26")


delay_stage = _delay_stage.DelayStage(stage_mot)
