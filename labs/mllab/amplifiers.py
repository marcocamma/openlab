from openlab.bme import BmeBox
from openlab.bme import AmplifierTiming
from . import config

try:
    box = BmeBox(ip="129.20.84.101",storage=config.storage,verbose=config.VERBOSE)

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
except Exception as e:
    print("Could define amplifier objects, error was",e)
    elite = usp = None

