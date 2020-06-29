import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import socket
import pickle

pg.setConfigOptions(antialias=True)

#server_address = ('localhost', 10000)
server_address = ('127.0.0.1', 10000)


class MainWindow(QtGui.QMainWindow):
    # initialize variables
    isStartButtonPressed = True 
    isStopButtonPressed = False
    
    def __init__(self):
        super(MainWindow, self).__init__()

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget(self)
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout()) # vertical layout
        
        pg.setConfigOption('background', 'w')
        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas) # add pyqtgraph widget

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label) # add label widget
             
        self.start_button = QtGui.QPushButton('start', self)
        self.start_button.clicked.connect(self.handleStartButton)
        self.mainbox.layout().addWidget(self.start_button) # add stop button
        
        self.stop_button = QtGui.QPushButton('stop', self)
        self.stop_button.clicked.connect(self.handleStopButton)
        self.mainbox.layout().addWidget(self.stop_button) # add stop button

        #  line plot
        self.plot = self.canvas.addPlot()    # add canvas for Plot
        self.plot.showGrid(x = True, y = True, alpha = 0.3) # show grid
        #self.plot.addLegend()
        #self.plot.setClipToView(True)
        #self.plot.setDownsampling(ds=1,auto=False,mode='peak')
        self.h2 = self.plot.plot(pen=pg.mkPen(color = (45, 89, 134),width = 2),name = "I")
        self.h3 = self.plot.plot(pen=pg.mkPen(color = (255, 102, 0),width = 2),name = "Q")
        #self.plot.setLabel('bottom','Time', units = 's')
        #self.plot.setLabel('left','Amplitude', units = 'V')
        #### Set Data  #####################
        self.counter = 0
        self.x = np.linspace(0,1e-5, num=100)
        self.y = np.zeros(len(self.x))
        self.x3 = np.linspace(0,1e-5, num=100)
        self.y3 = np.zeros(len(self.x3))

        #### Start  #####################
        self.refresh()

    def refresh(self):
        self.h2.setData(self.x,self.y)
        self.h3.setData(self.x3,self.y3)
        #self.plot.setDownsampling(auto=True,mode='subsample')
        #self.plot.setAutoVisible(True)
        tx = 'Counter:  {0:d}'.format(self.counter)
        self.label.setText(tx)
        self.counter += 1
        
    def handleStopButton(self):
        self.isStopButtonPressed = True
        
    def handleStartButton(self):
        self.isStartButtonPressed = True

class LivePlotThread(QtCore.QThread):

    def __init__(self , window): 
        QtCore.QThread.__init__(self) 
        self.window = window
        
        self.WantToCollectData = False    # initialize WantToCollectData flag
              
    def run(self):
        
        # read data
        while True:
            if self.window.isStartButtonPressed:
                sock = socket.socket()    # Create a TCP/IP socket
                sock.bind(server_address) # Bind the socket to the port
                sock.listen(1)    # start listening (maximum 1 connection)
                print('Listening for data...')
                self.WantToCollectData = True
                self.window.isStartButtonPressed = False
            if self.window.isStopButtonPressed:
                sock.close()
                print('Socket is closed.')
                self.WantToCollectData = False
                self.window.isStopButtonPressed = False
                self.window.counter = 0
            if self.WantToCollectData:
                c,a = sock.accept()
                data = b''
                while True:
                    block = c.recv(4096)
                    if not block: break
                    data += block
                c.close()
                unserialized_input = pickle.loads(data,encoding='bytes')
                self.window.x = np.asarray(unserialized_input[0])
                self.window.y = np.asarray(unserialized_input[1])
                self.window.x3 = np.asarray(unserialized_input[2])
                self.window.y3 = np.asarray(unserialized_input[3])
                self.window.refresh()

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))

window = MainWindow()
thread = LivePlotThread(window)
thread.start()
window.show()

sys.exit(app.exec_())