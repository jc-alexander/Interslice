import numpy as np

class spectr:
    """
    Dummy class for testing interslice waveforms
    """

    def __init__ (self):
        self.recsize = 1000
        self.nacq = 1
        self.nrec = 1
        self.awg = awg()
        self.dig = dig()

    def RecordSize(self,n = None):
        if n!=None: self.recsize = n
        return (self.recsize)

    def NumberOfAcquisitions(self,n = None):
        if n!=None: self.nacq = n
        return (self.nacq)
    def NumRecordsPerAcquisition(self,n = None):
        if n!=None: self.nrec = n
        return (self.nrec)


    def IQ_data_averaged(self, *args):
        print("run dummy IQ data averaged complete")
        return(np.zeros(int(self.recsize)),np.zeros(int(self.recsize)),np.linspace(0,self.recsize/self.dig.SR,int(self.recsize)),np.linspace(0,self.recsize/self.dig.SR,int(self.recsize)))

class dig:
    def __init__(self):
        self.navg=1
        self.SR = 1.6e9/(2**5)

    def SampleRate(self,n = None):
        if n!=None: self.SR = n
        #print("hello"+str(self.SR))
        return (self.SR)

    def NumberOfAverages(self,n = None):
        if n!=None: self.navg = n
        return (self.navg)


class ix:
    def __init__(self):
        self.x = 0
    def field(self,n = None):
        if n!=None: self.x = n
        return (self.x)

class iy:
    def __init__(self):
        self.y = 0
    def field(self,n = None):
        if n!=None: self.y = n
        return (self.y)

class iz:
    def __init__(self):
        self.z = 0
    def field(self,n = None):
        if n!=None: self.z = n
        return (self.z)


class field_3d:
    def __init__(self):
        self.r = 0
    def field(self,n = None):
        if n!=None: self.r = n
        return (self.r)

class awg:
    """
    dummy awg
    """
    def __init__ (self):
        self.ch12 = ch12()

    def generate_MarkersArray(self,*args):
        '''
        This method builds the final binary marker mask
        (bit position = destination bit position - 16)
        The following order is supposed:
        1st argument: gate marker (External1)
        2nd argument: acquisition marker (External2)
        3rd argument: digitizer marker (PXI_TRIG0)
        4th argument: user marker (PXI_TRIG1)
        '''
        _marker = np.zeros(len(args[0]))
        for i in range(len(args)):
            _marker = _marker + (2**i)*np.array(args[i])
        return _marker

    def ClearMemory(self):
        print("dummy memory cleared")



class ch12:
    def create_sequence(self, *args):
        print("sequence created")

    def init_channel(self, *args):
        print("dummy channel initialised")

    def load_waveform(self,*args):
        print("dummy waveform loaded")


class vsg:
    def __init__(self):
        self.x = 10e9
    def frequency(self,n = None):
        if n!=None: self.x = n

        return (self.x)
