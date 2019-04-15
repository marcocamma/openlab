from PyQt5 import QtGui,QtCore,QtWidgets
from pyqtgraph import ImageView
import numpy as np
#import pyqtgraph as pg


def get_img():
    return np.random.random( (1000,1000) )

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, camera = None):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.image_view = ImageView()
        #self.layout.addWidget(self.image_view)

        ## Create window with ImageView widget
        self.setCentralWidget(self.image_view)
        while True:
            try:
                self.image_view.setImage(get_img())
            except KeyboardInterrupt:
                break

    def update_image(self):
        pass


if __name__ == "__main__":
    import sys
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
