"""
Simple GUI to display (and analyze spectrometer data)
based on embedding_in_qt5_sgskip example
"""
import sys
import time
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QCheckBox,QLineEdit,QLabel


import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib import pyplot

import lmfit
from matplotlib.figure import Figure


<<<<<<< HEAD
class MakeUpSpectrometer:
    def __init__(self,integration_time=0.1):
        self.w = np.arange(500,1000)
        self.integration_time=integration_time

    def get_wavelengths(self): return self.w
    
    def set_integration_time(self,int_time):
        self.integration_time=int_time

    def acquire(self,naverage=1):
        amp = 100
        noise = 2
        w = self.w
        integration_time = self.integration_time
        bkg = np.cos(w/100)+100
        signal = amp*np.exp(-(w-800)**2/2/10**2)*integration_time
        noise = np.asarray( [np.random.normal(scale=noise,size=w.shape) for _ in range(naverage)]).mean(axis=0)
        i = bkg+signal+noise
        return i
=======
def makeup_data(integration_time=0.1,naverage=1,n=1000,amp=100,noise=2):
    w = np.arange(100,100+n)
    bkg = np.cos(w/100)+100
    signal = amp*np.exp(-(w-400)**2/2/10**2)*integration_time
    noise = np.asarray( [np.random.normal(scale=noise,size=n) for _ in range(naverage)]).mean(axis=0)
    i = bkg+signal+noise
    return w,i
>>>>>>> 44e193a8a1c405cbaf7c263e34a76727a237e349

def gaussfit(x,y):
    g = lmfit.models.GaussianModel()
    c = lmfit.models.ConstantModel()
    pars_g = g.guess(y,x=x)
    pars_c = c.guess(y[:5])
    model = g+c
    pars = pars_c+pars_g
    print(pars)
    res = model.fit(y,x=x,params=pars)
    return res


class MyDynamicMplCanvas(FigureCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, parent=None, width=5, height=4, dpi=100,
            get_data=None,frame_rate=1):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.grid()
        self.get_data=get_data

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

 
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        wait_time = 1/frame_rate
        wait_time_ms = wait_time*1e3
        timer.start(wait_time_ms)

    def compute_initial_figure(self):
        wavelenghts,intensities=self.get_data()
        self.axes.plot(wavelenghts,intensities)

    def update_figure(self):
        self.axes.lines[0].set_ydata(self.get_data()[1])
        self.draw()

    def plot_fit(self,res):
        x = res.userkws["x"]
        if len(self.axes.lines) == 1:
            self.axes.plot(x,res.best_fit)
        else:
            line = self.axes.lines[1]
            line.set_xdata(x)
            line.set_ydata(res.best_fit)
            



class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self,spectrometer):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.spectrometer = spectrometer

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&info', self.help)



        self.normalize=False
        self.integration_time = 1
        self.subtract_dark=False
        self.fit=False

        self.main_widget = QtWidgets.QWidget(self)



        input_pars_layout = QtWidgets.QHBoxLayout()
        cbox = QCheckBox("Normlize",self)
        cbox.stateChanged.connect(self.normalize_clickBox)
        input_pars_layout.addWidget(cbox)

        cbox = QCheckBox("Subtract Dark",self)
        cbox.stateChanged.connect(self.subtract_dark_clickBox)
        input_pars_layout.addWidget(cbox)

        cbox = QCheckBox("Fit",self)
        cbox.stateChanged.connect(self.fit_clickBox)
        input_pars_layout.addWidget(cbox)

        lab = QLabel("Integration time (s)",self)
        input_pars_layout.addWidget(lab)

        # Add text field
        ledit = QLineEdit(self)
        ledit.insert("0.1")
        self.integration_time_lineedit = ledit
        self.spectrometer.set_integration_time(0.1)
        input_pars_layout.addWidget(ledit)


        lab = QLabel("number of spectra to average",self)
        input_pars_layout.addWidget(lab)

        # Add text field
        ledit = QLineEdit(self)
        ledit.insert("3")
        self.naverage_lineedit = ledit
        input_pars_layout.addWidget(ledit)




        layout = QtWidgets.QVBoxLayout(self.main_widget)

        dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100,
                get_data=self.get_data)

        layout.addLayout(input_pars_layout)
        layout.addWidget(dc)
        self.plot_win = dc

        self.info_line = QLineEdit(self)
        self.info_line.insert("")
        self.info_line.move(350,20)
        self.info_line.resize(80,40)


        self.navi_toolbar = NavigationToolbar(dc, self)
        layout.addWidget(dc)  # the matplotlib canvas
        layout.addWidget(self.navi_toolbar)
        layout.addWidget(self.info_line)




        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)


    def get_data(self):
        try:
            integration_time = self.integration_time_lineedit.text()
            integration_time = float(integration_time)
            if integration_time != self.integration_time:
                self.spectrometer.set_integration_time(integration_time)
            self.integration_time = integration_time # store new value
        except ValueError:
            integration_time = self.integration_time

        try:
            nav = self.naverage_lineedit.text()
            nav = int(nav)
            self.naverage = nav
        except ValueError:
            nav = self.naverage

        w = self.spectrometer.get_wavelengths() 
        i = self.spectrometer.acquire(naverage=nav)
        if self.subtract_dark:
            #bkg = (i[0:10].mean() + i[-10:].mean())/2
            bkg = i[-10:].mean()
            i -= bkg
        if self.normalize:
            i /= i.max()
        if self.fit:
            t0 = time.time()
            wmin,wmax = self.plot_win.axes.get_xlim()
            idx = (w>wmin)&(w<wmax)
            wfit = w[idx]
            ifit = i[idx]
            res=gaussfit(wfit,ifit)
            self.plot_win.plot_fit(res)
            print(res.best_values.keys())
            fwhm = res.best_values['sigma']*2.35
            self.info_line.clear()
            info = "area = %.2e, center = %.1f nm, fwhm bw = %.1f nm"%(
                    res.best_values['amplitude'],
                    res.best_values['center'],
                    fwhm)

            self.info_line.insert(info)
            print(time.time()-t0)
        return w,i

    def fit_clickBox(self, state):
        self.fit = QtCore.Qt.Checked

    def normalize_clickBox(self, state):
        self.normalize = QtCore.Qt.Checked

    def subtract_dark_clickBox(self,state):
        self.subtract_dark = QtCore.Qt.Checked

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def help(self):
        QtWidgets.QMessageBox.about(self, "Help",
                                    """read spectrometer data and oftionally perform fit on data defined by visile xaxis """
                                )
        

if __name__ == "__main__":
    try:
        from openlab.oceanoptics import spectrometer
        s = spectrometer.Spectrometer()
    except:
<<<<<<< HEAD
        s = MakeUpSpectrometer()
=======
        get_data = makeup_data
>>>>>>> 44e193a8a1c405cbaf7c263e34a76727a237e349
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow(spectrometer=s)
    aw.setWindowTitle("Spectrometer simple GUI")
    aw.show()
    sys.exit(qApp.exec_())
#qApp.exec_()
