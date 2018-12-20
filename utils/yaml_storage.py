import collections
import pathlib
import os
import time
import shutil
import yaml

class Storage(collections.UserDict):
    """
    Class that adds saving to a dictionary. This happens automatically if 
    autosave is True
    """
    def __init__(self,*args,autosave=True,filename=None,**kwargs):
        print("Setting up storage, filename:",str(filename))

        self._autosave = autosave
        self._filename = filename
        super().__init__(self,*args,**kwargs)

        if filename is not None:
            self._filename = pathlib.Path(filename)
    
            if self._filename.is_file():
                self.read()
            else:
                self.data = {}
        else:
            self._autosave = False

    def __setitem__(self,key,value):
        super().__setitem__(key,value)
        if self._autosave: self.save()

    def __getitem__(self,key):
        if key in self.data:
            return self.data[key]
        else:
            return 0


    def read(self,fname=None):
        if fname is None:
            fname = self._filename
        else:
            fname = pathlib.Path(fname)

        with fname.open("r") as f:
            self.data = yaml.load(f)
        if self.data is None: self.data = dict()

    def save(self,fname=None,backup=False):

        if fname is None:
            fname = self._filename
        else:
            fname = pathlib.Path(fname)
            folder = fname.parent
            folder.mkdir(parents=True,exist_ok=True)

        if backup and fname.is_file():
            now = time.localtime()
            year = str(now.tm_year)
            now = time.strftime("%Y%m%d_%H%M%S",now)
            bck_fname = pathlib.Path(fname)
            bck_fname = bck_fname.parent / year / (now + "_" +bck_fname.name)
            bck_fname.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(str(fname),str(bck_fname))

        with fname.open("w") as f:
            yaml.dump(self.data,f)

