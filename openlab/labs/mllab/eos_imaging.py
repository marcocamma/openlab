import numpy as np
import openlab as ol
from openlab.generic.scan import ascan
from openlab.labs.mllab import thzsetup
from openlab.cameras import basler_gige
from . import config
import pathlib
import datastorage
from datastorage import DataStorage

storage = config.storage


#cam = basler_gige.Camera(ip="129.20.84.110")
cam = basler_gige.Camera()
cam.open()
#print("Going in hardware trigger")
#cam.camera.TriggerMode = "On"
#print("Camera hardware trigger",cam.camera.TriggerMode())

delay_stage = thzsetup.delay_stage

x = np.arange(0,1280)*5.3e-3
y = np.arange(0,1024)*5.3e-3

def acquire(nimages=10):
    if not cam.camera.IsOpen(): cam.camera.Open()
#    cam.camera.TriggerMode = "On"
    img = cam.get_images(nimages).mean(0)
    return dict(_x=x,_y=y,img=img)

def analyze(img):
    ol.utils.profile.analyze_2dprofile(x,y,img)


def scan_time(start,stop,N=100,nimages=20,folder="auto",comment=None,use_hw_trigger=False):
    """
    Parameters
    ----------
    folder : path or "auto"
        if auto uses variable defined in global storage (EOS_FOLDER field)
    """
    if use_hw_trigger:
        cam.hardware_trigger()
    else:
        cam.free_running()
    if folder == "auto":
        if not "EOS_IMAGING_FOLDER" in storage:
            raise RuntimeError("EOS_IMAGING_FOLDER not defined global storage, can't continue, please add or specify folder")
        else:
            folder = pathlib.Path(storage["EOS_IMAGING_FOLDER"])
    else:
        folder = pathlib.Path(folder)

    run = storage.get("EOS_RUN_NUMBER",0)+1
    storage['EOS_RUN_NUMBER'] = run
    storage.save()
    fname = folder / str("scan%04d.h5"%run)

    def _acquire():
        return acquire(nimages=nimages)
    if comment is None: comment = ""
    comment += "\naxis3 position = %.3f"%(thzsetup.xyz.z.wm())
    ascan(delay_stage,start,stop,N,acquire=_acquire,fname=fname,comment=comment)



def scan_focus(start, end, N=10):
    for zpos in np.linspace(start,end,N+1):
        thzsetup.xyz.z.move(zpos,wait=True)
        scan_time(-5,4,100,100,comment="50/50, 2 Ge filters + 1 card")

def scan_calibration(start,end,N):
    ZnTe_ypos = np.linspace(start, end, N+1)
    images =[]
    for ypos in ZnTe_ypos:
        thzsetup.xyz.y.move(ypos, wait = True)
        data = acquire(nimages = 1)
        images.append(data['img'])
    folder = pathlib.Path(storage["EOS_IMAGING_FOLDER"])
    fname = folder/str('calibration_scan.h5')
    to_save = dict(ZnTe_ypos = ZnTe_ypos, images = images)
    datastorage.save(fname, to_save)