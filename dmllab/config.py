import pathlib
from ..utils import yaml_storage

folder = pathlib.Path(__file__).parent
storage_fname = folder / "offsets.yaml"

VERBOSE=False

offset_storage = yaml_storage.Storage(filename=str(storage_fname), autosave=False)

