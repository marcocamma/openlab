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

class MakeUpCamera:
    def __init__(self,shape=(1000,1000),integration_time=0.1,noise=2):
        self.shape=shape
        self.integration_time=integration_time
        self.noise=noise
        self._x = np.arange(shape[0])

    def acquire(self,naverage=1):
        img = np.zeros(self.shape)
        for _ in range(naverage):
            y = np.cos(self._x/100)*self.integration_time
            img = [y for i in range(self.shape[1])]
            img = np.asarray(img)
            img += np.random.normal(scale=self.noise,size=self.shape)
        return img/naverage

    def set_exp_time(self,value):
        self.integration_time = value

    def set_gain(self,gain):
        self.gain = gain

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
        intensities=self.get_data()
        self.axes.imshow(self.get_data())

    def update_figure(self):
        self.axes.images[0].set_data(self.get_data())
        self.draw()

#    def plot_fit(self,res):
#        x = res.userkws["x"]
#        if len(self.axes.lines) == 1:
#            self.axes.plot(x,res.best_fit)
#        else:
#            line = self.axes.lines[1]
#            line.set_xdata(x)
#            line.set_ydata(res.best_fit)
            



class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self,camera):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.camera = camera

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&info', self.help)



        self.normalize=False
        self.subtract_dark = False
        self.integration_time = 1
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

        # INTEGRATION TIME
        lab = QLabel("Integration time (s)",self)
        input_pars_layout.addWidget(lab)
        ledit = QLineEdit(self)
        ledit.insert("0.1")
        self.integration_time_lineedit = ledit
        self.camera.set_exp_time(0.1)
        input_pars_layout.addWidget(ledit)

        # GAIN
        lab = QLabel("Gain",self)
        input_pars_layout.addWidget(lab)
        ledit = QLineEdit(self)
        ledit.insert("1")
        self.gain = 1
        self.gain_lineedit = ledit
        self.camera.set_gain(1)
        input_pars_layout.addWidget(ledit)



        # AVERAGE
        lab = QLabel("number of spectra to average",self)
        input_pars_layout.addWidget(lab)
        ledit = QLineEdit(self)
        ledit.insert("1")
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

    def show_info(self,info):
        if hasattr(self,"info_line"):
            self.info_line.clear()
            self.info_line.insert(info)

    def get_data(self):
        print("in get data")
        try:
            integration_time = self.integration_time_lineedit.text()
            integration_time = float(integration_time)
            if integration_time != self.integration_time:
                self.camera.set_exp_time(integration_time)
                self.show_info("")
                self.integration_time = integration_time # store new value
        except ValueError as err:
            integration_time = self.integration_time
            self.show_info(err.args[0])

        try:
            nav = self.naverage_lineedit.text()
            nav = int(nav)
            self.naverage = nav
            if hasattr(self,"info_line"): self.info_line.insert("")
        except ValueError:
            nav = self.naverage

        try:
            gain = self.gain_lineedit.text()
            gain = int(gain)
            if gain != self.gain:
                self.camera.set_gain(gain)
                self.gain = gain
                self.show_info("")
        except ValueError as err:
            gain = self.gain
            self.gain_lineedit.insert(str(gain))
            nav = self.naverage
            self.show_info(err.args[0])

        i = self.camera.acquire(naverage=nav)
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
        return i

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
        from openlab.cameras import basler_gige
        camera =basler_gige.Camera(num=0)
    except:
        camera = MakeUpCamera()
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow(camera=camera)
    aw.setWindowTitle("Camera simple GUI")
    aw.show()
    sys.exit(qApp.exec_())
#qApp.exec_()
