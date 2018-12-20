import pathlib
from openlab.utils import yaml_storage

folder = pathlib.Path(__file__).parent
storage_fname = folder / "storage.yaml"
VERBOSE=False

storage = yaml_storage.Storage(filename=str(storage_fname), autosave=True)

