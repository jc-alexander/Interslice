import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import socket
import pickle

pg.setConfigOptions(antialias=True)

#server_address = ('localhost', 10000)
server_address = ('127.0.0.3', 10000)


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
        

        #self.plot.enableAutoRange()
        self.plot.setClipToView(True)
        self.plot.addLegend(offset = -20)
        
        self.h5 = self.plot.plot(pen=pg.mkPen(color = (163, 0, 204),width = 5),name = "digitiser")
        self.h4 = self.plot.plot(pen=pg.mkPen(color = (64, 191, 128),width = 4),name = "acquisition")
        self.h3 = self.plot.plot(pen=pg.mkPen(color = (179, 0, 0),width = 3),name = "pulse gate")        
        self.h2 = self.plot.plot(pen=pg.mkPen(color = (255, 102, 0),width = 2),name = "Q")
        self.h1 = self.plot.plot(pen=pg.mkPen(color = (45, 89, 134),width = 2),name = "I")
        
        self.plot.setLabel('bottom','Time', units = 's')
        self.plot.setLabel('left','Amplitude', units = 'V')
        
        #### Set Data  #####################
        self.counter = 0
        self.x1 = np.linspace(0,1e-5, num=100)
        self.y1 = np.zeros(len(self.x1))
        self.x2 = np.linspace(0,1e-5, num=100)
        self.y2 = np.zeros(len(self.x2))
        self.x3 = np.linspace(0,1e-5, num=100)
        self.y3 = np.zeros(len(self.x2))
        self.x4 = np.linspace(0,1e-5, num=100)
        self.y4 = np.zeros(len(self.x2))
        self.x5 = np.linspace(0,1e-5, num=100)
        self.y5 = np.zeros(len(self.x2))

        #### Start  #####################
        self.refresh()

    def refresh(self):
        self.h1.setData(self.x1,self.y1)
        self.h2.setData(self.x2,self.y2)
        self.h3.setData(self.x3,self.y3)
        self.h4.setData(self.x4,self.y4)
        self.h5.setData(self.x5,self.y5)
        self.plot.setDownsampling(auto=True,mode='subsample')
        self.plot.setAutoVisible(True)
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
                u_in = pickle.loads(data,encoding='bytes')
                self.window.x1  = np.linspace(0,len(u_in[0])/u_in[5],len(u_in[0]))
                self.window.y1  = np.asarray(u_in[0])
                self.window.x2 = np.linspace(0,len(u_in[1])/u_in[5],len(u_in[1]))
                self.window.y2 = np.asarray(u_in[1])
                self.window.x3 = np.linspace(0,len(u_in[2])/u_in[5],len(u_in[2]))
                self.window.y3 = np.asarray(u_in[2])
                self.window.x4 = np.linspace(0,len(u_in[3])/u_in[5],len(u_in[3]))
                self.window.y4 = np.asarray(u_in[3])
                self.window.x5 = np.linspace(0,len(u_in[4])/u_in[5],len(u_in[4]))
                self.window.y5 = np.asarray(u_in[4])
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