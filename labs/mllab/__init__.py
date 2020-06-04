try:
    from .amplifiers import usp
    from .amplifiers import elite
except Exception  as e:
    print("Could define amplifiers objects, error was '%s'"%e)

from . import thzsetup
from . import eos_scan
from . import pyro
from . import Ge
from .thzsetup import scope
from .thzsetup import delay_stage
from . import shg
from . import scanpyroelectric

try:
    from . import eos_imaging
except Exception as e:
    print("thzsetup: Could not import EOS imaging module, error was",e)
