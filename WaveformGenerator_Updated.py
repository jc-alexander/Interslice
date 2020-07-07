# Import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.integrate import cumtrapz

class WaveformGenerator:
    """
    Class containing all functions required to create pulsed and sine waveforms
    """

    quantum  = 4 # waveforms should be muptiple of the quantum (see length correction)
    minimum_sample_number = 12 # minimum number of points in a waveform array
    sample_rate = 1.28e9 # sample rate
    gate_delay = 10e-9 # gate delay for high power amplifier

    def delay(self,duration):
        data = np.zeros(int(round(duration*self.sample_rate)))
        return data

    def pulse(self,amplitude, duration):
        data = amplitude * np.ones(int(round(duration*self.sample_rate)))
        return data

    def combine(self,*args):
        data = [] # define data array
        for arg in args:
            data = np.append(data, arg)
        return data

    def marker(self,*args):
        '''
        This method creates marker array using start and duration times (absolute)
        provided in the format [start,duration]:
            create_gate_marker([start1,duration1],[start2,duration2])
        '''
        _marker = np.array([])
        _absolute_end = 0
        for i in range(len(args)):
            start = args[i][0]
            duration = args[i][1]
            _marker = self.combine(_marker,
                                   self.delay(start - _absolute_end),
                                   self.pulse(1, duration))
            _absolute_end = start + duration
        return _marker

    def rect_pulses(self,*args):
        '''
        This method creates rectangular pulses array using amlitude,
        start and duration times (absolute) provided
        in the format [amplitude,start,duration]:
            create_gate_marker([amp1,start1,duration1],[amp2,start2,duration2])
        '''
        _pulse = np.array([])
        _absolute_end = 0
        for i in range(len(args)):
            amplitude = args[i][0]
            start = args[i][1]
            duration = args[i][2]
            _pulse = self.combine(_pulse,
                                  self.delay(start - _absolute_end),
                                  self.pulse(amplitude, duration))
            _absolute_end = start + duration
        return _pulse

    def duration(self,data):
        return len(data)/self.sample_rate

    def length_correction(self,*args):
        '''
        This method corrects length of input arrays
        '''
        # determine the maximum number of points in the initial arrays
        max_len = 0
        for i in range(len(args)):
            if len(args[i])>max_len:
                max_len = len(args[i])
        # correct length
        data = list(args)
        for i in range(len(data)):
            # make all arrays to be the same length
            data[i] = np.append(data[i], np.zeros(max_len - len(data[i])))

            # make lengths of all arrays to be multiple of the quantum by
            # discarding last points
            n = len(data[i]) % self.quantum
            if n!=0:
                data[i] = data[i][:-n]

            # make the length larger than the minimum size
            if len(data[i])<self.minimum_sample_number:
                data[i] = np.append(data[i], np.zeros(self.minimum_sample_number - len(data[i])))
        return data

    def plot(self,*args):
        '''
        This method plots input data
        '''
        plot = plt.figure()
        for i in range(len(args)):
            plt.plot(self.time(args[i])/1e-9, args[i])
            plt.xlabel('Time (ns)')
            plt.ylabel('Amplitude (a.u.)')
        return plot

    def sine(self,amplitude, frequency, phase, duration, dc_offset):
        number_of_points = int(duration*self.sample_rate)
        data = np.zeros(number_of_points) # define data array
        time = np.linspace(0,(number_of_points-1)/self.sample_rate,number_of_points) # time array
        data = dc_offset + amplitude*np.sin((2*np.pi*frequency*time)+phase)
        return data

    def cos(self,amplitude, frequency, phase, duration, dc_offset):
        number_of_points = int(duration*self.sample_rate)
        data = np.zeros(number_of_points) # define data array
        time = np.linspace(0,(number_of_points-1)/self.sample_rate,number_of_points) # time array
        data = dc_offset + amplitude*np.cos((2*np.pi*frequency*time)+phase)
        return data

    def heterodyne_pulse(self,amplitude,duration,phase):
        data = amplitude * np.ones(int(round(duration*self.sample_rate)))
        return ([data,phase])

    def phaseramp_pulse(self,amplitude,duration,phase0,phase1):
        data = amplitude * np.ones(int(round(duration*self.sample_rate)))
        phase = np.linspace(phase0,phase1,int(round(duration*self.sample_rate)))
        return ([data,phase])

    def heterodyne_gaussian_pulse(self,amplitude,duration,phase,standard_deviation=None):
        #duration is time for FWHM of pulse, unless standard deviation is specified
        n_samples = int(round(2*duration*self.sample_rate)) # Clip gaussian to double FWHM

        if not standard_deviation:
            #2.3548 conversion factor bethwen FWHM and SD
            standard_deviation = duration/2.3548*self.sample_rate

        data = amplitude * gaussian(n_samples,standard_deviation)
        return ([data,phase])

    def heterodyne_delay(self,duration, phase):
        data = np.zeros(int(round(duration*self.sample_rate)))
        return ([data,phase])

    def heterodyne_combine(self,*args,w,t0=0):
        I = []
        Q = []
        LO = []
        for pulse in args:
            t = self.time(pulse[0],self.sample_rate)
            I = np.append(I,np.multiply(np.sin(w*(t+t0)+np.pi*pulse[1]/180),pulse[0]))
            Q = np.append(Q,np.multiply(np.cos(w*(t+t0)+np.pi*pulse[1]/180),pulse[0]))

            t0 = t0+len(pulse[0])/self.sample_rate
        return Q,I

    def add_noise (self,wave,level):
        noise = np.random.normal(0,level,int(len(wave)/100))
        noise = np.resize(np.repeat(noise,100, axis=0),len(wave))
        return(np.clip(np.add(wave,noise),-1,1))

    #def add_noise_polar (self,wave,level):
    #    noise = np.random.normal(0,level,int(len(wave)/100))
    #    noise = np.repeat(noise,100, axis=1)
    #    return([np.clip(np.add(wave[0],noise),-1,1),wave[1]])

    def heterodyne_multi_combine(self,*args, repeats):
        data = []
        for i in range (int(repeats)):
            for pulse in args:
                data.append(pulse)
        return data

    def time(self,data,*arg):
        if len(arg) == 0:
            sample_rate = self.sample_rate
        else:
            sample_rate = arg[0]
        return np.linspace(0, (len(data)-1)/sample_rate, len(data))

    def save_waveform(self,filename, data):
        np.savetxt(filename, data)
        print('Saved waveform data to file ',filename)

    def load_waveform(self,filename):
        data = [] # define data array
        data = np.loadtxt(filename)
        return data

    def multi_combine(self,data1, data2, repeats):
        data = []
        multi_data = np.append(data1, data2)
        for i in range (int(repeats)):
            data = np.append(data, multi_data)
        return data

    def pulse_gen(self,I_array, Q_array, t, w, phase):
        y = np.sin(w*t + phase)
        z = np.cos(w*t + phase)
        I_mod = np.multiply(y, I_array)
        Q_mod = np.multiply(z, Q_array)
        return I_mod, Q_mod

    def signal_demod(self,I_mod, Q_mod, t, w):
        demodulated_I1 = I_mod*np.cos(w*t)
        demodulated_Q1 = Q_mod*np.sin(w*t)
        demodulated_I2 = I_mod*np.sin(w*t)
        demodulated_Q2 = Q_mod*np.cos(w*t)
        I_out = demodulated_I1-demodulated_Q1
        Q_out = demodulated_I2+demodulated_Q2
        return I_out, Q_out

    def wurst_pulse(self,dw,phase,duration,WURSTN=20, amp=1):

        points=int(duration*self.sample_rate)
        t = np.linspace(0, duration, points)

        dt=t[1]-t[0]
        deltaw=dw
        startw = -dw/2
        tcent=duration/2
        AM = 1 - abs(np.sin((np.pi*(t-tcent))/duration))**WURSTN
        FM = startw+ deltaw/duration * t

        I=AM*np.cos(2*np.pi*cumtrapz(FM, initial=0)*dt+np.pi*phase/180)*amp
        Q=AM*np.sin(2*np.pi*cumtrapz(FM, initial=0)*dt+np.pi*phase/180)*amp

        return I, Q

    def heterodyne_wurst_pulse(self,dw,phase,duration,WURSTN=20, amp=1):

        points=int(duration*self.sample_rate)
        t = np.linspace(0, duration, points)

        dt=t[1]-t[0]
        deltaw=dw
        startw = -dw/2
        tcent=duration/2
        AM = 1 - abs(np.sin((np.pi*(t-tcent))/duration))**WURSTN
        FM = startw+ deltaw/duration * t

        return ([AM*amp,360*cumtrapz(FM, initial=0)*dt+phase])

    def heterodyne_wurst_arctan(self,dw,phase,duration,WURSTN=20, amp=1, linearity = 0.2):

        points=int(duration*self.sample_rate)
        t = np.linspace(0, duration, points)

        dt=t[1]-t[0]
        deltaw=dw
        startw = -dw/2
        tcent=duration/2
        AM = 1 - abs(np.sin((np.pi*(t-tcent))/duration))**WURSTN
        FM = np.arctan((t-t[-1]/2)/(linearity*np.median(t)))
        FM = dw/2*FM/FM.max()

        return ([AM*amp,360*cumtrapz(FM, initial=0)*dt+phase])

    def heterodyne_wurst_sinh(self,dw,phase,duration, WURSTN=20, amp=1, nonlinearity = 3):

        points=int(duration*self.sample_rate)
        t = np.linspace(0, duration, points)

        dt=t[1]-t[0]
        deltaw=dw
        startw = -dw/2
        tcent=duration/2
        AM = 1 - abs(np.sin((np.pi*(t-tcent))/duration))**WURSTN
        FM = np.sinh(np.linspace(-nonlinearity,nonlinearity,points))
        FM = dw/2*FM/FM.max()

        return ([AM*amp,360*cumtrapz(FM, initial=0)*dt+phase])

    def wurst_theta(self,w_start, w_stop,phase,duration,WURSTN=20, amp=1):

        points=int(duration*self.sample_rate)
        t = np.linspace(0, duration, points)

        dt=t[1]-t[0]
        deltaw=w_stop-w_start
        tcent=duration/2
        AM = 1 - abs(np.sin((np.pi*(t-tcent))/duration))**WURSTN
        FM = w_start+ deltaw/duration * t

        I=AM*np.cos(2*np.pi*cumtrapz(FM, initial=0)*dt+phase)*amp
        Q=AM*np.sin(2*np.pi*cumtrapz(FM, initial=0)*dt+phase)*amp

        return I, Q

    def bir(self,theta,chirp,duration,lamb=10,beta=np.arctan(10),amp=1):
        # This is a BIR-4 pulse sequence for arbitrary theta rotation
        # Ref: Handbook of MRI pulse sequences, Sec. 6.1.4
        # Adapted from Matlab software "MRiLab"
        #
        # theta is intended rotation angle
        # chirp is total frequency rang to chirp over
        # t is time vector for entire sequences
        # lamb is lambda parameter for pulse envelope modulation
        # beta is parameter for chirp modulation

        npts = int(duration*self.sample_rate)
        t = np.linspace(0, duration, npts)

        half_time = duration/2;
        quarter_time = duration/4;
        dt = 1/self.sample_rate

        # Four pulse sequence
        time1 = np.array([i for i in t if i < quarter_time])
        amp1 = 1*np.tanh(lamb*(1-4*np.divide(time1,duration)))
        freq1 = np.tan(np.multiply(beta,4*np.divide(time1,duration)))/(np.tan(beta)*2*np.pi)

        time2 = np.array([i for i in t if i >= quarter_time and i < half_time])
        amp2 = 1*np.tanh(lamb*(4*np.divide(time2,duration)-1))
        freq2 = np.tan(np.multiply(beta,4*np.divide(time2,duration)-2))/(np.tan(beta)*2*np.pi)

        time3 = np.array([i for i in t if i >= half_time and i < 3*quarter_time])
        amp3 = 1*np.tanh(lamb*(3-4*np.divide(time3,duration)))
        freq3 = np.tan(np.multiply(beta,4*np.divide(time3,duration)-2))/(np.tan(beta)*2*np.pi)

        time4 = np.array([i for i in t if i >= 3*quarter_time])
        amp4 = 1*np.tanh(lamb*(4*np.divide(time4,duration)-3))
        freq4 = np.tan(np.multiply(beta,4*np.divide(time4,duration)-4))/(np.tan(beta)*2*np.pi)

        phi1 =  np.pi + theta/2 # Discontinuity in phase determines theta rotation
        phi2 = -np.pi - theta/2

        freqshift = 0.5*chirp/max(freq1) # +/- frequency to chirp over
        freq = 2*np.pi*np.multiply(freqshift,np.concatenate([freq1, freq2, freq3, freq4]))
        # This freq is in rad/s

        phase = cumtrapz(freq,initial=0)*dt # Integrate frequency to get accumulated phase
        phase = np.add(phase,np.concatenate([np.repeat(0,len(time1)), np.repeat(phi1,len(time2)+len(time3)), np.repeat(phi1+phi2,len(time4))]))

        amplitude=np.multiply(amp,np.concatenate([amp1, amp2, amp3, amp4]))

        I = amplitude * np.cos(phase)
        Q = amplitude * np.sin(phase)

        #amplitude = amplitude
        #freq = freq/(2*np.pi)
        #phase = phase
        #time = t

        return I,Q

    def birwurst(self,theta,chirp,duration,wurst_n=20,beta=np.arctan(10),amp=1,rise_and_fall=True):
        # This is a BIR-4 pulse sequence for arbitrary theta rotation
        # Ref: Handbook of MRI pulse sequences, Sec. 6.1.4
        # Adapted from Matlab software "MRiLab"
        # Envelope modified to match that of WURST pulse
        #
        # theta is intended rotation angle
        # chirp is total frequency rang to chirp over
        # t is time vector for entire sequences
        # wurstn is parameter for pulse envelope modulation
        # beta is parameter for chirp modulation
        # rise_and_fall determines whether to include ramp at start and end of sequence

        npts = int(duration*self.sample_rate)
        t = np.linspace(0, duration, npts)

        half_time = duration/2;
        quarter_time = duration/4;
        dt = 1/self.sample_rate

        # Four pulse sequence
        time1 = np.array([i for i in t if i < quarter_time])
        freq1 = np.tan(np.multiply(beta,4*np.divide(time1,duration)))/(np.tan(beta)*2*np.pi)

        time2 = np.array([i for i in t if i >= quarter_time and i < half_time])
        freq2 = np.tan(np.multiply(beta,4*np.divide(time2,duration)-2))/(np.tan(beta)*2*np.pi)

        time3 = np.array([i for i in t if i >= half_time and i < 3*quarter_time])
        freq3 = np.tan(np.multiply(beta,4*np.divide(time3,duration)-2))/(np.tan(beta)*2*np.pi)

        time4 = np.array([i for i in t if i >= 3*quarter_time])
        freq4 = np.tan(np.multiply(beta,4*np.divide(time4,duration)-4))/(np.tan(beta)*2*np.pi)

        phi1 =  np.pi + theta/2 # Discontinuity in phase determines theta rotation
        phi2 = -np.pi - theta/2

        freqshift = 0.5*chirp/max(freq1) # +/- frequency to chirp over
        freq = 2*np.pi*np.multiply(freqshift,np.concatenate([freq1, freq2, freq3, freq4]))
        # This freq is in rad/s

        phase = cumtrapz(freq,initial=0)*dt # Integrate frequency to get accumulated phase
        phase = np.add(phase,np.concatenate([np.repeat(0,len(time1)), np.repeat(phi1,len(time2)+len(time3)), np.repeat(phi1+phi2,len(time4))]))

        amplitude=np.multiply(amp,1-abs(np.sin(np.pi*t/half_time))**wurst_n);
        if rise_and_fall:
            e1 = (1-abs(np.sin(np.pi*(time1-quarter_time)/half_time))**wurst_n)
            e2 = np.squeeze(np.ones([1,len(time2)+len(time3)]))
            e3 = (1-abs(np.sin(np.pi*(time4-quarter_time)/half_time))**wurst_n)
            envelope = np.concatenate((e1, e2, e3))
            amplitude = np.multiply(amplitude,envelope);

        I = amplitude * np.cos(phase)
        Q = amplitude * np.sin(phase)

        #amplitude = amplitude
        #freq = freq/(2*np.pi)
        #phase = phase
        #time = t

        return I,Q

    def plot_waveforms(self,I,Q,marker_gate,marker_acq,SR):
        t_HE = 1e6*np.linspace(0,len(I)/SR,len(I))
        plt.plot(t_HE,I,label = 'I')
        plt.plot(t_HE,Q,label = 'Q')
        plt.plot(t_HE,marker_gate,label = 'gate')
        plt.plot(t_HE,marker_acq,label = 'acqisition')
        plt.legend()
        plt.xlabel('Time ($\mu$s)')
        plt.ylabel('Amplitude')
        plt.show()

    def bump_pulse(self,amplitude,duration,phase,k,k_c,standard_deviation=None):
        #Pulse From Probst's Paper to Reduce Resonator Ringing

        n_samples = int(round(duration*self.sample_rate))
        time = np.linspace(-duration/2,duration/2,n_samples)
        data = []
        t_c = 0 #Centre of Pulse
        for t in time:
            x = (t-t_c)/(duration/2) #Variable x to make things easier

            bump = amplitude*np.exp(-1/(abs(1-(x)**2))) #Bump Function
            diff = amplitude*(-2*((np.exp(1))**(1/((x**2)-1)))*(x)/(1-(x)**2)**2)*(2/duration) # Differential of Bump Function

            pulse = (diff + (k/2)*bump)/np.sqrt(k_c) #Pulse sent to resonator
            data.append(pulse)

        data = amplitude*(np.array(data)/np.max(data[1:-1])) #first and last values u=in data are inf - Normalised to 1
        data[0]=data[-1]=0

        return ([data,phase])

    def BIP(self,duration,phase_params,time_intervals,amp):
        points=int(duration*self.sample_rate)
        t = np.linspace(-duration,duration, points)
        normalise_t_intervals = np.multiply(time_intervals,-1)#np.divide(time_intervals,time_intervals[-1])

        t_intervals_n = np.multiply(time_intervals[::-1],-1)
        phase_params_n = phase_params[::-1]
        normalise_t_intervals = np.concatenate((t_intervals_n[:-1],time_intervals))
        total_phase_params = np.concatenate((phase_params_n,phase_params))

        tau = ((t/duration))*10
        cubic_vec = [1,tau,np.square(tau),np.power(tau,3)]
        #phase_params = np.concatenate((phase_params,phase_params),axis=0)
        interval_index = []
        phis = np.array([])

        for i in range(len(normalise_t_intervals)):
            interval_index.append(np.argmin(abs(tau-normalise_t_intervals[i])))

        for i in range(len(interval_index)-1):
            tau_slice = abs(tau[interval_index[i]:interval_index[i+1]])
            cubic_vec = [np.power(tau_slice,0),tau_slice,np.square(tau_slice),np.power(tau_slice,3)]
            phi = np.dot(total_phase_params[i][:],cubic_vec)
            phis = np.concatenate((phis,phi))

        times = (tau[:-1]/(10/duration) + duration)/2
        amp = (np.zeros(len(times))+1)*amp

        return ([amp,phis])

    def hetrodyne_BIR(self,theta,chirp,duration,lamb=10,beta=np.arctan(10),amp=1):

        npts = int(duration*self.sample_rate)

        npts = int(duration*self.sample_rate)
        t = np.linspace(0, duration, npts)

        half_time = duration/2;
        quarter_time = duration/4;
        dt = 1/self.sample_rate

        # Four pulse sequence
        time1 = np.array([i for i in t if i < quarter_time])
        amp1 = 1*np.tanh(lamb*(1-4*np.divide(time1,duration)))
        freq1 = np.tan(np.multiply(beta,4*np.divide(time1,duration)))/(np.tan(beta)*2*np.pi)

        time2 = np.array([i for i in t if i >= quarter_time and i < half_time])
        amp2 = 1*np.tanh(lamb*(4*np.divide(time2,duration)-1))
        freq2 = np.tan(np.multiply(beta,4*np.divide(time2,duration)-2))/(np.tan(beta)*2*np.pi)

        time3 = np.array([i for i in t if i >= half_time and i < 3*quarter_time])
        amp3 = 1*np.tanh(lamb*(3-4*np.divide(time3,duration)))
        freq3 = np.tan(np.multiply(beta,4*np.divide(time3,duration)-2))/(np.tan(beta)*2*np.pi)

        time4 = np.array([i for i in t if i >= 3*quarter_time])
        amp4 = 1*np.tanh(lamb*(4*np.divide(time4,duration)-3))
        freq4 = np.tan(np.multiply(beta,4*np.divide(time4,duration)-4))/(np.tan(beta)*2*np.pi)

        phi1 =  np.pi + theta/2 # Discontinuity in phase determines theta rotation
        phi2 = -np.pi - theta/2

        freqshift = 0.5*chirp/max(freq1) # +/- frequency to chirp over
        freq = 2*np.pi*np.multiply(freqshift,np.concatenate([freq1, freq2, freq3, freq4]))
        # This freq is in rad/s

        phase = cumtrapz(freq,initial=0)*dt # Integrate frequency to get accumulated phase
        phase = np.add(phase,np.concatenate([np.repeat(0,len(time1)), np.repeat(phi1,len(time2)+len(time3)), np.repeat(phi1+phi2,len(time4))]))

        amplitude=np.multiply(amp,np.concatenate([amp1, amp2, amp3, amp4]))

        return ([amplitude,phase*180/np.pi])
