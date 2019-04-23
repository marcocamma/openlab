import time
import numpy as np
from pypylon import pylon
import cv2

def get_camera(ip=None,num=0):
    """ if ip is not given, num is used the select the camera number """
    factory = pylon.TlFactory.GetInstance()
    if ip is None:
        cameras = factory.EnumerateDevices()
        camera = cameras[num]
        camera_device = factory.CreateDevice(camera)
        camera = pylon.InstantCamera(camera_device)
        #camera = pylon.InstantCamera(factory.CreateFirstDevice())
    else:
        ptl = factory.CreateTl('BaslerGigE')
        empty_camera_info = ptl.CreateDeviceInfo()
        empty_camera_info.SetPropertyValue('IpAddress', ip)
        camera_device = factory.CreateDevice(empty_camera_info)
        camera = pylon.InstantCamera(camera_device)
    try:
        camera.Open()
        print("Opening ",camera.DeviceModelName())
    except:
        pass
    return camera

class Camera(object):
    def __init__(self,ip=None,num=0,strategy=None):
        """ if ip is not given, num is used the select the camera number """
        self.camera = get_camera(ip=ip,num=num)
        if strategy is None: strategy = pylon.GrabStrategy_LatestImageOnly
        self.strategy = strategy
        #print(self.strategy)
        self._get_info()

    def set_gain(self,value):
        try:
            self.camera.GainRaw.SetValue(value)
        except Exception as err:
            raise ValueError(err.args[0])

    def get_gain(self):
        return self.camera.GainRaw()

    gain = property(get_gain,set_gain)

    def set_exp_time(self,value):
        try:
            value_us = int(value*1e6)
            self.camera.ExposureTimeAbs.SetValue(value_us)
        except Exception as err:
            raise ValueError(err.args[0])

    def get_exp_time(self):
        value_us = self.camera.ExposureTimeAbs()
        value = float(value_us/1e6)
        return value
    exp_time = property(get_exp_time,set_exp_time)

    def get_format(self):
        camera = self.camera
        return camera.PixelFormat.GetValue()

    def show_valid_format(self):
        camera = self.camera
        return camera.PixelFormat.Symbolics

    def set_format(self,value):
        if value not in valid_format:
            ans = "Allowed pixel format for camera are %s"%str(valid_format)
            raise ValueError(ans)
        else:
            camera.PixelFormat.SetValue(value)
    format = property(get_format,set_format)

    def get_image(self,timeout=5000,as_array=True):
        image  = self.camera.GrabOne(timeout)
        #image = converter.Convert(result)
        if as_array:
            image = image.GetArray()
        return image

    def acquire(self,naverage=1):
        images = np.asarray([self.get_image() for _ in range(naverage)])
        return images.mean(axis=0)

    def get_images_old(self,timeout=5000):
        camera = self.camera
        n=0
        t0=time.time()
        if not camera.IsGrabbing():
            camera.StartGrabbing(self.strategy)
        try:
            while camera.IsGrabbing():
                image = camera.RetrieveResult(timeout)
                if image.GrabSucceeded():
                    # Access the image data
                    #image = converter.Convert(grabResult)
                    image = image.GetArrayZeroCopy()
                    #image = image.GetArray()
                    dt = time.time()-t0
                    n += 1
                    rate = n/dt
                    print("rate",rate)
                    yield image
                    #print(image.sum())
        except KeyboardInterrupt:
            pass
        finally:
            camera.StopGrabbing()


    def get_images(self,timeout=5000):
        camera = self.camera
        n=0
        t0=time.time()
        if not camera.IsGrabbing():
            camera.StartGrabbing(self.strategy)
        try:
            while camera.IsGrabbing():
                with camera.RetrieveResult(timeout) as result:
                    with result.GetArrayZeroCopy() as image:
                        dt = time.time()-t0
                        n += 1
                        rate = n/dt
                        print("rate",rate)
                        yield image
        except KeyboardInterrupt:
            pass
        finally:
            camera.StopGrabbing()

    def _get_info(self):
        camera = self.camera
        info = camera.GetDeviceInfo()
        ret = dict()
        mne = info.GetUserDefinedName()
        if mne != "": ret['name'] = mne
        mac = info.GetMacAddress()
        mac = ':'.join(mac[i:i+2] for i in range(0, len(mac), 2))
        ret['mac'] = mac
        ret['ip'] = info.GetIpAddress()
        ret['model'] = info.GetModelName()
        self.info = ret
        return self.info

    def show_info(self):
        fields = "name model ip mac".split()
        s = []
        for f in fields:
            if f in self.info:
                s.append("%s = %s"%(f,self.info[f]))
        s.append("format = %s"%self.get_format())
        return "\n".join(s)

    def get_str(self,full=False):
        info = self.info
        if "name" in info:
            s = info["name"] + "(" + ret["model"] +")"
        else:
            s = info["model"]
        if full:
            s += ", ip %s" % self.info["ip"]
            s += ", format %s" % self.format
        return s


    def __str__(self):
        return self.get_str(full=True)

    def __repr__(self):
        name = self.info.get("name",self.info["model"])
        return name+"("+self.info["ip"]+")"

    def display(self):
        camera = self.camera
        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000,
                    pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data
                image = converter.Convert(grabResult)
                img = image.GetArrayZeroCopy() #Array()
                cv2.namedWindow('title', cv2.WINDOW_NORMAL)
                cv2.imshow('title', img)
                k = cv2.waitKey(1)
                if k == 27:
                    break
            grabResult.Release()
        camera.StopGrabbing()
        cv2.destroyAllWindows()
