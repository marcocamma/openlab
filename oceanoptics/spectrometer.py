"""
Small layer on top of seabreeze to display data and do some simple analysis
"""

import seabreeze.spectrometers as sb
from matplotlib import pyplot as plt
import numpy as np

class Spectrometer:
    def __init__(self,device=None,integration_time=20e-3):
        if device is None: device = sb.Spectrometer(sb.list_devices()[0])
        self.device = device
        self.w = device.wavelengths()
        self.set_integration_time(integration_time)

    def get_wavelengths(self): return self.w

    def set_integration_time(self,integration_time=20e-3):
        """
        Parameters
        ----------
        integration_time : float
            integration_time in sec
        """
        integration_time_us = integration_time*1e6
        try:
            self.device.integration_time_micros(integration_time_us)
        except:
            pass

    def acquire(self,naverage=1):
        """ TODO: do it with async """
        i = np.zeros_like(self.w)
        for _ in range(naverage):
            i += self.device.intensities()
        return i/naverage


    def display(self,wait_time=0.2,averaging=1,integration_time=None):
        """
        Parameters
        ----------
        rate : float
            refresh rate in Hz
        """
        if integration_time is not None:
            self.set_integration_time(integration_time)
        data = self.acquire(averaging)
        line, = plt.plot(self.w,data)
        try:
            while True:
                data = self.acquire(averaging)
                line.set_ydata(data)
                plt.pause(wait_time)
        except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    spec = Spectrometer()
    spec.display()
