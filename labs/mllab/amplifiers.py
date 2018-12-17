from ..bme import BmeBox
from ..bme import AmplifierTiming
from . import config

box = BmeBox(ip="129.20.76.14",storage=config.offset_storage,verbose=config.VERBOSE)

if box is None:
    elite = usp = None
else:
    elite = AmplifierTiming(
        amplifier_name="elite",
        amplifier_card=box.dg1,
        burst_card=box.dg2)

    usp =   AmplifierTiming(
        amplifier_name="usp",
        amplifier_card=box.dg3,
        burst_card=box.dg4)

