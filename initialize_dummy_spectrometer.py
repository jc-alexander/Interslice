import numpy as np
from time import time
from time import sleep
import datetime
import matplotlib.pyplot as plt
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from WaveformGenerator_Updated import WaveformGenerator
from resonator_tools import *
from dummy_spectrometer import *

spectr = spectr()
wfg = WaveformGenerator()
sgs = vsg()
ix = ix()
iy = iy()
iz = iz()
i3d = field_3d()

def ABBA_CPMG(wait,amp_exc,amp_A,amp_B,t_exc,tA,tB,fA,fB,shot_rep,python_delay,NB,SR = 1.28e9/(2**2),
             phase_exc = 0,phaseA = 0,phaseB=0,python_avgs = 1,pickle_input = True,pickle_output = True,
             save= False, name = "",folder = "C:\\Users\\Administrator\\Documents\\",
             saveraw = False,window = 12):

    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)

    piebytwo     = wfg.heterodyne_pulse(amp_exc,t_exc,phase_exc)
    piebytwo_neg = wfg.heterodyne_pulse(amp_exc,t_exc,phase_exc+180)

    d2 = wfg.heterodyne_delay(wait,0)

    wurstA = wfg.heterodyne_wurst_pulse(fA,phaseA,tA,amp=amp_A)
    wurstB = wfg.heterodyne_wurst_pulse(fB,phaseB,tB,amp=amp_B)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST inversion pulse A and B

    wurstA_I,wurstA_Q   = wfg.heterodyne_combine(d1, wurstA,d2,d2,w = w,t0=0)
    wurstA_I_neg,wurstA_Q_neg = -wurstA_I,-wurstA_Q

    wurstB_I,wurstB_Q   = wfg.heterodyne_combine(d1, wurstB,d2,d2,w = w,t0=0)
    wurstB_I_neg,wurstB_Q_neg = -wurstB_I,-wurstB_Q

    gate_len_A = len(wurstA_I)/SR -2*wait+150e-9
    gate_len_B = len(wurstB_I)/SR -2*wait+150e-9
    acq_wait = 0

    wurst_gate_A      = wfg.pulse(1,gate_len_A)
    wurst_gate_B      = wfg.pulse(1,gate_len_B)
    wurst_acq       = wfg.delay(gate_len_A)
    wurst_digitizer = wfg.delay(gate_len_A)

    A_pos = create_waveform("A_pos",wurstA_I    , wurstA_Q    , wurst_gate_A, wurst_acq,wurst_digitizer)
    A_neg = create_waveform("A_neg",wurstA_I_neg, wurstA_Q_neg, wurst_gate_A, wurst_acq,wurst_digitizer)

    B_pos = create_waveform("B_pos",wurstB_I    , wurstB_Q    , wurst_gate_B, wurst_acq,wurst_digitizer)
    B_neg = create_waveform("B_neg",wurstB_I_neg, wurstB_Q_neg, wurst_gate_B, wurst_acq,wurst_digitizer)

    ###############################################################
    #Excitation waveform x
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d1,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d1,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,t_exc/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.pulse(1,1e-6)

    x_pos = create_waveform("x_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    x_neg = create_waveform("x_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)
    ###############################################################
    # delay waveform for reptime setting
    end_delay = 1e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    reset = create_waveform("reset",*[wfg.delay(end_delay)]*4,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)
    N1 = int(wait/end_delay)
    N2 = int(shot_rep/end_delay)

    if NB == 0:
        SE_p = ['SE_p',[x_pos,delay,A_pos,reset,delay],
                       [1    ,N1   ,2    ,1    ,N2   ]]
        spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

        SE_n = ['SE_n',[x_neg,delay,A_pos,reset,delay],
                       [1    ,N1   ,2    ,1    ,N2   ]]
    else:

        SE_p = ['SE_p',[x_pos,delay,A_pos,B_pos,A_pos,reset,delay],
                       [1    ,N1   ,1    ,NB    ,1    ,1    ,N2   ]]
        spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

        SE_n = ['SE_n',[x_neg,delay,A_pos,B_pos,A_pos,reset,delay],
                       [1    ,N1   ,1    ,NB    ,1    ,1    ,N2   ]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        seq = SE_n
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        seq = SE_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def ABBA_arctan(Pi,wait,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,shot_rep,python_delay,SR = 1.28e9/(2**2),
             inversion_wurst_pi=None,phase = 0,python_avgs = 1,pickle_input = True,pickle_output = True,
             save= False, name = "",folder = "C:\\Users\\Administrator\\Documents\\",
             saveraw = False,window = 12):

    title2 = "Without pre-inversion"

    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_cart = wfg.delay(150e-9)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)
    d2_cart = wfg.delay(wait)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    wurst = wfg.heterodyne_wurst_arctan(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    wurstB = wfg.heterodyne_wurst_arctan(-wurst_freq,phase,wurst_Pi,amp=wurst_amp)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST inversion pulse A and B

    wurst_I,wurst_Q   = wfg.heterodyne_combine(d1, wurst,d1,w = w,t0=0)
    wurst_I_neg,wurst_Q_neg = -wurst_I,-wurst_Q

    wurstB_I,wurstB_Q   = wfg.heterodyne_combine(d1, wurstB,d1,w = w,t0=0)
    wurstB_I_neg,wurstB_Q_neg = -wurstB_I,-wurstB_Q

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    A_pos = create_waveform("A_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    A_neg = create_waveform("A_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    B_pos = create_waveform("B_pos",wurstB_I    , wurstB_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    B_neg = create_waveform("B_neg",wurstB_I_neg, wurstB_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #Excitation waveform x
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d1,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d1,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.pulse(1,1e-6)

    x_pos = create_waveform("x_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    x_neg = create_waveform("x_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)
    ###############################################################
    #Excitation waveform y
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d1,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d1,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.pulse(0,1e-6)

    y_pos = create_waveform("y_pos",Q_in_pos,-I_in_pos, marker_gate,marker_acq,marker_digitizer)
    y_neg = create_waveform("y_neg",Q_in_neg,-I_in_neg, marker_gate,marker_acq,marker_digitizer)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 1e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    reset = create_waveform("reset",*[wfg.delay(end_delay)]*4,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)
    N1 = int(wait/end_delay)
    N2 = int(0.5*wait/end_delay)
    N3 = int(shot_rep/end_delay)

    SE_p = ['SE_p',[x_pos,delay,A_pos,delay,y_pos,delay,B_pos,delay,B_pos,delay,A_pos,reset,delay],
                   [1    ,N2   ,1    ,N2   ,1    ,N2   ,1    ,N1   ,1    ,N1   ,1    ,1    ,N3   ]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[x_neg,delay,A_pos,delay,y_neg,delay,B_pos,delay,B_pos,delay,A_pos,reset,delay],
                   [1    ,N2   ,1    ,N2   ,1    ,N2   ,1    ,N1   ,1    ,N1   ,1    ,1    ,N3   ]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        seq = SE_n
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        seq = SE_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def AABB_amp(Pi,wait,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,shot_rep,python_delay,SR = 1.28e9/(2**2),
             phase = 0,python_avgs = 1,pickle_input = True,pickle_output = True,
             save= False, name = "",folder = "C:\\Users\\Administrator\\Documents\\",
             saveraw = False,window = 12):


    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_cart = wfg.delay(150e-9)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)
    d2_cart = wfg.delay(wait)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    wurst = wfg.heterodyne_wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    wurstB = wfg.heterodyne_wurst_pulse(-wurst_freq,phase,wurst_Pi,amp=wurst_amp)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST inversion pulse A and B

    wurst_I,wurst_Q   = wfg.heterodyne_combine(d1, wurst,d1,w = w,t0=0)
    wurst_I_neg,wurst_Q_neg = -wurst_I,-wurst_Q

    wurstB_I,wurstB_Q   = wfg.heterodyne_combine(d1, wurstB,d1,w = w,t0=0)
    wurstB_I_neg,wurstB_Q_neg = -wurstB_I,-wurstB_Q

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    A_pos = create_waveform("A_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    A_neg = create_waveform("A_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    B_pos = create_waveform("B_pos",wurstB_I    , wurstB_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    B_neg = create_waveform("B_neg",wurstB_I_neg, wurstB_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #Excitation waveform x
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d1,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d1,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.pulse(1,1e-6)

    x_pos = create_waveform("x_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    x_neg = create_waveform("x_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)
    ###############################################################
    #Excitation waveform y
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d1,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d1,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.pulse(0,1e-6)

    y_pos = create_waveform("y_pos",Q_in_pos,-I_in_pos, marker_gate,marker_acq,marker_digitizer)
    y_neg = create_waveform("y_neg",Q_in_neg,-I_in_neg, marker_gate,marker_acq,marker_digitizer)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 1e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    reset = create_waveform("reset",*[wfg.delay(end_delay)]*4,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)
    N1 = int(wait/end_delay)
    N2 = int(0.5*wait/end_delay)
    N3 = int(shot_rep/end_delay)

    SE_p = ['SE_p',[x_pos,delay,A_pos,delay,y_pos,delay,A_pos,delay,B_pos,delay,B_pos,reset,delay],
                   [1    ,N2   ,1    ,N2   ,1    ,N2   ,1    ,N1   ,1    ,N1   ,1    ,1    ,N3   ]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[x_neg,delay,A_pos,delay,y_neg,delay,A_pos,delay,B_pos,delay,B_pos,reset,delay],
                   [1    ,N2   ,1    ,N2   ,1    ,N2   ,1    ,N1   ,1    ,N1   ,1    ,1    ,N3   ]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        seq = SE_n
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        seq = SE_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def ABBA_amp(Pi,wait,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,shot_rep,python_delay,SR = 1.28e9/(2**2),
             phase = 0,python_avgs = 1,pickle_input = True,pickle_output = True,
             save= False, name = "",folder = "C:\\Users\\Administrator\\Documents\\",
             saveraw = False,window = 12):


    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_cart = wfg.delay(150e-9)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)
    d2_cart = wfg.delay(wait)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    wurst = wfg.heterodyne_wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    wurstB = wfg.heterodyne_wurst_pulse(-wurst_freq,phase,wurst_Pi,amp=wurst_amp)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST inversion pulse A and B

    wurst_I,wurst_Q   = wfg.heterodyne_combine(d1, wurst,d1,w = w,t0=0)
    wurst_I_neg,wurst_Q_neg = -wurst_I,-wurst_Q

    wurstB_I,wurstB_Q   = wfg.heterodyne_combine(d1, wurstB,d1,w = w,t0=0)
    wurstB_I_neg,wurstB_Q_neg = -wurstB_I,-wurstB_Q

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    A_pos = create_waveform("A_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    A_neg = create_waveform("A_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    B_pos = create_waveform("B_pos",wurstB_I    , wurstB_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    B_neg = create_waveform("B_neg",wurstB_I_neg, wurstB_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #Excitation waveform x
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d1,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d1,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.pulse(1,1e-6)

    x_pos = create_waveform("x_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    x_neg = create_waveform("x_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)
    ###############################################################
    #Excitation waveform y
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d1,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d1,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.pulse(0,1e-6)

    y_pos = create_waveform("y_pos",Q_in_pos,-I_in_pos, marker_gate,marker_acq,marker_digitizer)
    y_neg = create_waveform("y_neg",Q_in_neg,-I_in_neg, marker_gate,marker_acq,marker_digitizer)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 1e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    reset = create_waveform("reset",*[wfg.delay(end_delay)]*4,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)
    N1 = int(wait/end_delay)
    N2 = int(0.5*wait/end_delay)
    N3 = int(shot_rep/end_delay)

    SE_p = ['SE_p',[x_pos,delay,A_pos,delay,y_pos,delay,B_pos,delay,B_pos,delay,A_pos,reset,delay],
                   [1    ,N2   ,1    ,N2   ,1    ,N2   ,1    ,N1   ,1    ,N1   ,1    ,1    ,N3   ]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[x_neg,delay,A_pos,delay,y_neg,delay,B_pos,delay,B_pos,delay,A_pos,reset,delay],
                   [1    ,N2   ,1    ,N2   ,1    ,N2   ,1    ,N1   ,1    ,N1   ,1    ,1    ,N3   ]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        seq = SE_n
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        seq = SE_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def silenced_echo_unbalanced(Pi,wait,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,long_wait,shot_rep,python_delay,SR = 1.28e9/(2**2),
                  inversion_wurst_pi=None,phase = 0,python_avgs = 1,invert = False, pickle_input = True,pickle_output = True,
                  save= False,inversion_wurst_amp = None, wait2 = None,name = "",folder = "C:\\Users\\Administrator\\Documents\\",
                  saveraw = False,N_inv = 1,d_inv = 150e-9,window = 12,refpulse_amp = 0, wurst_amp2 = None):
    if refpulse_amp==0: refpulse_gate=0
    else: refpulse_gate = 1
    if inversion_wurst_pi==None: inversion_wurst_pi = wurst_Pi
    if inversion_wurst_amp == None: inversion_wurst_amp = wurst_amp
    if wurst_amp2 == None: wurst_amp2 = wurst_amp
    if wait2==None: wait2 = 2*wait

    if invert == True:
        title2 = "With pre-inversion"
        name = name+"inverted"

    else: title2 = "Without pre-inversion"

    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_cart = wfg.delay(150e-9)
    d_inv_cart = wfg.delay(d_inv)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)


    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    wurst2 = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp2)
    wurst_inversion = wfg.wurst_pulse(wurst_freq,phase,inversion_wurst_pi,amp=inversion_wurst_amp)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I = wfg.combine(d1_cart,wurst2[0],d1_cart)
    wa_Q = wfg.combine(d1_cart,wurst2[1],d1_cart)

    wa_I_neg = wfg.combine(d1_cart,-wurst2[0],d1_cart)
    wa_Q_neg = wfg.combine(d1_cart,-wurst2[1],d1_cart)

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST inversion pulse

    wurst_I     = wfg.combine(d1_cart,wurst[0],d1_cart)
    wurst_Q     = wfg.combine(d1_cart,wurst[1],d1_cart)

    wurst_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wurst_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #WURST inversion pulse

    wurst_inversion_I     = wfg.combine(d1_cart,wurst_inversion[0],d_inv_cart)
    wurst_inversion_Q     = wfg.combine(d1_cart,wurst_inversion[1],d_inv_cart)

    wurst_inversion_I_neg = wfg.combine(d1_cart,-wurst_inversion[0],d_inv_cart)
    wurst_inversion_Q_neg = wfg.combine(d1_cart,-wurst_inversion[1],d_inv_cart)

    gate_len = 150e-9-d_inv+len(wurst_inversion_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_inversion_pos = create_waveform("wurst_inversion_pos",wurst_inversion_I    , wurst_inversion_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_inversion_neg = create_waveform("wurst_inversion_neg",wurst_inversion_I_neg, wurst_inversion_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # delay waveform for inter-pulse delay
    short_delay = 1e-6
    ip_delay = create_waveform("ip_delay",*[wfg.delay(short_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    r1 = wfg.delay(10e-6)
    r2 = wfg.pulse(refpulse_amp,10e-6)
    r3 = wfg.pulse(refpulse_gate,10e-6)
    reset = create_waveform("reset",r2,r1,r3,r1,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)

    SE_inv_p = ['SE_inv_p',[wurst_inversion_pos,delay,In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_inv_n = ['SE_inv_n',[wurst_inversion_pos,delay,In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

    SE_p = ['SE_p',[In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        if invert == True: seq = SE_inv_n
        else: seq = SE_n
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        if invert == True: seq = SE_inv_p
        else: seq = SE_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def silenced_multi_echo(t_exc,wait_exc,wait_ref,N_exc,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,long_wait,shot_rep,python_delay,
                        SR = 1.28e9/(2**2),inversion_wurst_pi=100e-6,phase = 0,python_avgs = 1,invert = False,
                        pickle_waveforms = True,save= False,inversion_wurst_amp = None, name = "",
                        folder = "C:\\Users\\Administrator\\Documents\\",saveraw = False,window = 12):

    if inversion_wurst_amp == None: inversion_wurst_amp = wurst_amp
    inter_wurst_delay = (2*N_exc+1)*(wait_exc+t_exc)+2*wait_ref+wait_exc


    if invert == True:
        title2 = "With pre-inversion"
        name = name+"_inverted"

    else: title2 = "Without pre-inversion"

    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_cart = wfg.delay(150e-9)

    p_exc = wfg.heterodyne_pulse(pulse_amplitude,t_exc,phase)
    p_exc_neg= wfg.heterodyne_pulse(pulse_amplitude,t_exc,phase+180)

    p90_exc = wfg.heterodyne_pulse(pulse_amplitude,t_exc,phase+90)
    p90_exc_neg= wfg.heterodyne_pulse(pulse_amplitude,t_exc,phase+270)

    d2 = wfg.heterodyne_delay(wait_exc,0)

    wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    wurst_inversion = wfg.wurst_pulse(wurst_freq,phase,inversion_wurst_pi,amp=inversion_wurst_amp)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I = wfg.combine(d1_cart,wurst[0],d1_cart)
    wa_Q = wfg.combine(d1_cart,wurst[1],d1_cart)

    wa_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wa_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait_ref-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST refocusing pulse

    wurst_I     = wfg.combine(d1_cart,wurst[0],d1_cart)
    wurst_Q     = wfg.combine(d1_cart,wurst[1],d1_cart)

    wurst_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wurst_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #WURST inversion pulse

    wurst_inversion_I     = wfg.combine(d1_cart,wurst_inversion[0],d1_cart)
    wurst_inversion_Q     = wfg.combine(d1_cart,wurst_inversion[1],d1_cart)

    wurst_inversion_I_neg = wfg.combine(d1_cart,-wurst_inversion[0],d1_cart)
    wurst_inversion_Q_neg = wfg.combine(d1_cart,-wurst_inversion[1],d1_cart)

    gate_len = len(wurst_inversion_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_inversion_pos = create_waveform("wurst_inversion_pos",wurst_inversion_I    , wurst_inversion_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_inversion_neg = create_waveform("wurst_inversion_neg",wurst_inversion_I_neg, wurst_inversion_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,p_exc    ,d2,p90_exc    ,d2,w = w,t0=0)
    I_in_neg,Q_in_neg =  wfg.heterodyne_combine(d1,p_exc_neg,d2,p90_exc_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,t_exc+200e-9),wfg.delay(wait_exc-200e-9),wfg.pulse(1,t_exc+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)
    ################################################################
    #Excitation waveform
    I_s_pos,Q_s_pos =  wfg.heterodyne_combine(d1,p90_exc    ,d2,w = w,t0=0)
    I_s_neg,Q_s_neg =  wfg.heterodyne_combine(d1,p90_exc_neg,d2,w = w,t0=0)

    gate_len = len(I_s_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,t_exc+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    s_pos = create_waveform("s_pos",-I_s_pos, -Q_s_pos, marker_gate,marker_acq,marker_digitizer)
    s_neg = create_waveform("s_neg",-I_s_neg, -Q_s_neg, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # delay waveform for inter-pulse delay
    short_delay = 1e-6
    ip_delay = create_waveform("ip_delay",*[wfg.delay(short_delay)]*5)
    ###############################################################
    # initial trigger for digitiser
    r1 = wfg.delay(100e-9)
    r2 = wfg.pulse(1,100e-9)
    trig = create_waveform("trig",*[r1]*4,r2)
    ###############################################################
    # end trigger for digitiser reset
    r1 = wfg.delay(10e-6)
    r2 = wfg.pulse(0,10e-6)
    r3 = wfg.pulse(0,10e-6)
    reset = create_waveform("reset",r2,r1,r3,r1,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)

    N1 = int(long_wait/end_delay)
    N2 = int(wait_ref/short_delay)
    N3 = int(inter_wurst_delay/short_delay)
    N4 = int(shot_rep/end_delay)

    SE_inv_p = ['SE_inv_p',[wurst_inversion_pos,delay,trig,In_pos,s_pos,ip_delay,wurst_pos,ip_delay,wurst_pos,reset,delay],
               [1,N1,1,N_exc,1,N2,1,N3,1,1,N4]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_inv_n = ['SE_inv_n',[wurst_inversion_pos,delay,trig,In_neg,s_neg,ip_delay,wurst_pos,ip_delay,wurst_pos,reset,delay],
                [1,N1,1,N_exc,1,N2,1,N3,1,1,N4]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

    SE_p = ['SE_p',[trig,In_pos,s_pos,ip_delay,wurst_pos,ip_delay,wurst_pos,reset,delay], [1,N_exc,1,N2,1,N3,1,1,N4]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[trig,In_neg,s_neg,ip_delay,wurst_pos,ip_delay,wurst_pos,reset,delay],[1,N_exc,1,N2,1,N3,1,1,N4]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()
        if invert == True: seq = SE_inv_n
        else: seq = SE_n
        if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.3") #Pickling a very long sequence can take a long time! Deactivate when not needed

        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

        start_time = time()
        if invert == True: seq = SE_inv_p
        else: seq = SE_p
        if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.2") #Pickling a very long sequence can take a long time! Deactivate when not needed
        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"_silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"_silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"_silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"_silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"_silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()

        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def square_train(amp1,amp2,t1,t2,phase1,phase2,reps1,reps2,wait,shot_rep,python_delay=0,
                 python_avgs=1,pickle_waveforms=True,save= False,SR=1.28e9/(2**2),w=0,phasecycle=True,window=12,
                 acq_len = 300e-6,acq_wait = 'auto',folder = "C:\\Users\\Administrator\\Documents\\",
                 name = "",pickle_output = True,phase1_neg = None,phase2_neg = None, plot_output = True):

    if phase1_neg == None: phase_neg = phase
    if phase2_neg == None: phase_neg = phase+180

    wfg.sample_rate = SR
    gate_wait12 = 50e-9

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)

    p1     = wfg.heterodyne_pulse(amp1,t1,phase1)
    p1_neg = wfg.heterodyne_pulse(amp1,t1,phase1_neg)

    p2     = wfg.heterodyne_pulse(amp2,t2,phase2)
    p2_neg = wfg.heterodyne_pulse(amp2,t2,phase2_neg)

    d2 = wfg.heterodyne_delay(wait,0)

    dt = wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')
    ###############################################################
    #Square pulse with wait
    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,p1    ,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,p1_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait12),wfg.pulse(1,t1+200e-9))

    marker_acq = wfg.combine(wfg.delay(gate_len))
    marker_digitizer = wfg.combine(wfg.delay(gate_len))

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)
    ###############################################################
    #Square pulse with wait and acquisition
    I_in_pos_acq,Q_in_pos_acq = wfg.heterodyne_combine(d1,p2    ,d2,w = w,t0=0)
    I_in_neg_acq,Q_in_neg_acq = wfg.heterodyne_combine(d1,p2_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos_acq)/SR
    if acq_wait == "auto": acq_wait = 0

    marker_gate = wfg.combine(wfg.delay(gate_wait12),wfg.pulse(1,t2+200e-9))

    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    In_pos_acq = create_waveform("In_pos_acq",I_in_pos_acq, Q_in_pos_acq, marker_gate,marker_acq,marker_digitizer)
    In_neg_acq = create_waveform("In_neg_acq",I_in_neg_acq, Q_in_neg_acq, marker_gate,marker_acq,marker_digitizer)
    ################################################################
    # end trigger for digitiser reset
    reset = create_waveform("reset",*[wfg.delay(10e-6)]*4,wfg.combine(dd,dt))
    ###############################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    AWG_auto_setup(SR)

    SE_p = ['SE_p', [In_pos,In_pos_acq,reset,delay], [reps1,reps2,1,int(shot_rep/end_delay)] ]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n', [In_neg,In_neg_acq,reset,delay], [reps1,reps2,1,int(shot_rep/end_delay)] ]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        if phasecycle == True:
            start_time = time()
            seq = SE_n
            if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.3") #Pickling a very long sequence can take a long time! Deactivate when not needed
            while (time()-start_time)<(python_delay-0.05): sleep(0.1)
            spectr.awg.ch12.init_channel(seq[0])
            n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
            n_Is.append(n_I)
            n_Qs.append(n_Q)
            n_I = np.mean(n_Is,axis = 0)
            n_Q = np.mean(n_Qs,axis = 0)
            print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

        start_time = time()
        seq = SE_p
        if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.2") #Pickling a very long sequence can take a long time! Deactivate when not needed
        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        p_Is.append(p_I)
        p_Qs.append(p_Q)
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        if phasecycle == True:
            PhasedI = np.subtract(p_I, n_I)
            PhasedQ = np.subtract(p_Q, n_Q)
        else:
            PhasedI = p_I
            PhasedQ = p_Q
        #Demodulate from intermediate carrier frequency
        t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
        I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        I = downsample(I_demod,window)
        Q = downsample(Q_demod,window)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot

        if pickle_output == True: plot_pickle(t/1e9,I,Q)
        current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5
        if phasecycle == True: print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        else: print('Phase cycling off, average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        if save==True:
            np.savetxt(folder+"\\"+name+"HahnEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,sgs.frequency()*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"HahnEcho.pdf")
            plt.close()

    if plot_output == True:plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,sgs.frequency()*1e-6))
    return(t,I,Q,mag)

def sinc_square(amp1,amp2,t1,t2,phase1,phase2,wait12,frequency1,shot_rep,python_delay=0,
                   python_avgs=1,pickle_waveforms=True,save= False,SR=1.28e9/(2**2),w=0,phasecycle=True,window=12,
                   acq_len = 300e-6,acq_wait = 'auto',folder = "C:\\Users\\Administrator\\Documents\\",
                   name = "",pickle_output = True,phase1_neg = None,phase2_neg = None, plot_output = True):

    if phase1_neg == None: phase1_neg = phase1+180
    if phase2_neg == None: phase2_neg = phase2

    wfg.sample_rate = SR
    gate_wait12 = 50e-9

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)

    piebytwo     = wfg.heterodyne_sinc_pulse(amp1,frequency1,t1,phase1)
    piebytwo_neg = wfg.heterodyne_sinc_pulse(amp1,frequency1,t1,phase1_neg)

    d2 = wfg.heterodyne_delay(wait12,0)

    pie     = wfg.heterodyne_pulse(amp2,t2,phase2)
    pie_neg = wfg.heterodyne_pulse(amp2,t2,phase2_neg) # leave identical to pie unless you want to phase cycle the pi pulse!

    dt = wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo    ,d2,pie    ,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,pie_neg,w = w,t0=0)

    gate_len = len(I_in_pos)/SR
    if acq_wait == "auto": acq_wait = gate_len+wait12-20e-6

    g1=g2=1
    if amp1 ==0: g1 = 0
    if amp2 ==0: g2 = 0

    marker_gate = wfg.combine(wfg.delay(gate_wait12),wfg.pulse(g1,t1+200e-9),wfg.delay(wait12-200e-9),
                              wfg.pulse(g2,t2+200e-9))

    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,acq_len))
    marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,acq_len),dd,dt)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    AWG_auto_setup(SR)

    SE_p = ['SE_p', [In_pos,delay], [1,int(shot_rep/end_delay)] ]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n', [In_neg,delay], [1,int(shot_rep/end_delay)] ]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        if phasecycle == True:
            start_time = time()
            seq = SE_n
            if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.3") #Pickling a very long sequence can take a long time! Deactivate when not needed
            while (time()-start_time)<(python_delay-0.05): sleep(0.1)
            spectr.awg.ch12.init_channel(seq[0])
            n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
            n_Is.append(n_I)
            n_Qs.append(n_Q)
            n_I = np.mean(n_Is,axis = 0)
            n_Q = np.mean(n_Qs,axis = 0)
            print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

        start_time = time()
        seq = SE_p
        if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.2") #Pickling a very long sequence can take a long time! Deactivate when not needed
        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        p_Is.append(p_I)
        p_Qs.append(p_Q)
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        if phasecycle == True:
            PhasedI = np.subtract(p_I, n_I)
            PhasedQ = np.subtract(p_Q, n_Q)
        else:
            PhasedI = p_I
            PhasedQ = p_Q
        #Demodulate from intermediate carrier frequency
        t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
        I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        I = downsample(I_demod,window)
        Q = downsample(Q_demod,window)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot

        if pickle_output == True: plot_pickle(t/1e9,I,Q)
        current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5
        if phasecycle == True: print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        else: print('Phase cycling off, average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        if save==True:
            np.savetxt(folder+"\\"+name+"HahnEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,sgs.frequency()*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"HahnEcho.pdf")
            plt.close()

    if plot_output == True:plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,sgs.frequency()*1e-6))
    return(t,I,Q,mag)

def two_pulse_echo(amp1,amp2,t1,t2,phase1,phase2,wait12,shot_rep,python_delay=0,
                   python_avgs=1,pickle_waveforms=True,save= False,SR=1.28e9/(2**2),w=0,phasecycle=True,window=12,
                   acq_len = 300e-6,acq_wait = 'auto',folder = "C:\\Users\\Administrator\\Documents\\",
                   name = "",pickle_output = True,phase1_neg = None,phase2_neg = None, plot_output = True):

    if phase1_neg == None: phase1_neg = phase1+180
    if phase2_neg == None: phase2_neg = phase2

    wfg.sample_rate = SR
    gate_wait12 = 50e-9

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)

    piebytwo     = wfg.heterodyne_pulse(amp1,t1,phase1)
    piebytwo_neg = wfg.heterodyne_pulse(amp1,t1,phase1_neg)

    d2 = wfg.heterodyne_delay(wait12,0)

    pie     = wfg.heterodyne_pulse(amp2,t2,phase2)
    pie_neg = wfg.heterodyne_pulse(amp2,t2,phase2_neg) # leave identical to pie unless you want to phase cycle the pi pulse!

    dt = wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo    ,d2,pie    ,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,pie_neg,w = w,t0=0)

    gate_len = len(I_in_pos)/SR
    if acq_wait == "auto": acq_wait = gate_len+wait12-20e-6

    g1=g2=1
    if amp1 ==0: g1 = 0
    if amp2 ==0: g2 = 0

    marker_gate = wfg.combine(wfg.delay(gate_wait12),wfg.pulse(g1,t1+200e-9),wfg.delay(wait12-200e-9),
                              wfg.pulse(g2,t2+200e-9))

    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,acq_len))
    marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,acq_len),dd,dt)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    AWG_auto_setup(SR)

    SE_p = ['SE_p', [In_pos,delay], [1,int(shot_rep/end_delay)] ]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n', [In_neg,delay], [1,int(shot_rep/end_delay)] ]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        if phasecycle == True:
            start_time = time()
            seq = SE_n
            if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.3") #Pickling a very long sequence can take a long time! Deactivate when not needed
            while (time()-start_time)<(python_delay-0.05): sleep(0.1)
            spectr.awg.ch12.init_channel(seq[0])
            n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
            n_Is.append(n_I)
            n_Qs.append(n_Q)
            n_I = np.mean(n_Is,axis = 0)
            n_Q = np.mean(n_Qs,axis = 0)
            print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

        start_time = time()
        seq = SE_p
        if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.2") #Pickling a very long sequence can take a long time! Deactivate when not needed
        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        p_Is.append(p_I)
        p_Qs.append(p_Q)
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        if phasecycle == True:
            PhasedI = np.subtract(p_I, n_I)
            PhasedQ = np.subtract(p_Q, n_Q)
        else:
            PhasedI = p_I
            PhasedQ = p_Q
        #Demodulate from intermediate carrier frequency
        t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
        I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        I = downsample(I_demod,window)
        Q = downsample(Q_demod,window)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot

        if pickle_output == True: plot_pickle(t/1e9,I,Q)
        current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5
        if phasecycle == True: print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        else: print('Phase cycling off, average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        if save==True:
            np.savetxt(folder+"\\"+name+"HahnEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,sgs.frequency()*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"HahnEcho.pdf")
            plt.close()

    if plot_output == True:plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,sgs.frequency()*1e-6))
    return(t,I,Q,mag)

def three_pulse_echo(amp1,amp2,amp3,t1,t2,t3,phase1,phase2,phase3,wait12,wait23,shot_rep,python_delay=0,
                   python_avgs=1,pickle_waveforms=True,save= False,SR=1.28e9/(2**2),w=0,phasecycle=True,window=12,
                   acq_len = 300e-6,acq_wait = 'auto',folder = "C:\\Users\\Administrator\\Documents\\",
                   name = "",pickle_output = True,phase1_neg = None,phase2_neg = None,phase3_neg = None):

    if phase1_neg == None: phase1_neg = phase1+180
    if phase2_neg == None: phase2_neg = phase2
    if phase3_neg == None: phase3_neg = phase3

    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5
    frequency = sgs.frequency()
    wfg.sample_rate = SR
    gate_wait = 50e-9

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d01 = wfg.heterodyne_delay(150e-9,0)

    p1     = wfg.heterodyne_pulse(amp1,t1,phase1)
    p1_neg = wfg.heterodyne_pulse(amp1,t1,phase1_neg)

    d12 = wfg.heterodyne_delay(wait12,0)

    p2     = wfg.heterodyne_pulse(amp2,t2,phase2)
    p2_neg = wfg.heterodyne_pulse(amp2,t2,phase2_neg)

    d23 = wfg.heterodyne_delay(wait23,0)

    p3     = wfg.heterodyne_pulse(amp3,t3,phase3)
    p3_neg = wfg.heterodyne_pulse(amp3,t3,phase3_neg)

    dt = wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d01,p1    ,d12,p2    ,d23,p3    ,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d01,p1_neg,d12,p2_neg,d23,p3_neg,w = w,t0=0)

    gate_len = len(I_in_pos)/SR
    if acq_wait == "auto": acq_wait = gate_len+wait12-20e-6

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,t1+200e-9),wfg.delay(wait12-200e-9),
                              wfg.pulse(1,t2+200e-9),wfg.delay(wait23-200e-9),wfg.pulse(1,t3+200e-9))

    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,acq_len))
    marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,acq_len),dd,dt)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    AWG_auto_setup(SR)

    SE_p = ['SE_p', [In_pos,delay], [1,int(shot_rep/end_delay)] ]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n', [In_neg,delay], [1,int(shot_rep/end_delay)] ]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        if phasecycle == True:
            start_time = time()
            seq = SE_n
            if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.3") #Pickling a very long sequence can take a long time! Deactivate when not needed
            while (time()-start_time)<(python_delay-0.05): sleep(0.1)
            spectr.awg.ch12.init_channel(seq[0])
            n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
            n_Is.append(n_I)
            n_Qs.append(n_Q)
            n_I = np.mean(n_Is,axis = 0)
            n_Q = np.mean(n_Qs,axis = 0)
            print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

        start_time = time()
        seq = SE_p
        if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.2") #Pickling a very long sequence can take a long time! Deactivate when not needed
        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        p_Is.append(p_I)
        p_Qs.append(p_Q)
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        if phasecycle == True:
            PhasedI = np.subtract(p_I, n_I)
            PhasedQ = np.subtract(p_Q, n_Q)
        else:
            PhasedI = p_I
            PhasedQ = p_Q
        #Demodulate from intermediate carrier frequency
        t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
        I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        I = downsample(I_demod,window)
        Q = downsample(Q_demod,window)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot

        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if phasecycle == True: print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        else: print('Phase cycling off, average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        if save==True:
            np.savetxt(folder+"\\"+name+"_3PulseEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"_3PulseEcho.pdf")
            plt.close()

    plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def pickle_sequence(seq, SR, address = "127.0.0.2"):
    start_time = time()
    seq_full = [np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)]
    for m in range(len(seq[1])-1):
        waveform_name = seq[1][m][0]
        reps = seq[2][m]
        seq_full = np.concatenate((seq_full,*[seq[1][m][1:]]*reps), axis = 1)
    print("sequence built in %.3fs"%(time()-start_time))
    pickle_waveforms(*seq_full,SR,address = address)

def pickle_waveforms(y1,y2,y3,y4,y5,SR, address = "127.0.0.2"):
    start_time = time()
    server_address = (address, 10000)
    if len(y1)>100000:
        window = int(len(y1)/100000)
        y1 = downsample_simple(y1,window)
        y2 = downsample_simple(y2,window)
        y3 = downsample_simple(y3,window)
        y4 = downsample_simple(y4,window)
        y5 = downsample_simple(y5,window)
        SR = SR/window
    sock = socket.socket()
    sock.connect(server_address)
    serialized_data = pickle.dumps([np.array(y1),np.array(y2),np.array(y3),np.array(y4),np.array(y5),SR], protocol=2)
    sock.sendall(serialized_data)
    sock.close()
    print("waveforms pickled in %.3fs"%(time()-start_time))

def plot_waveforms(I,Q,marker_gate,marker_acq,marker_dig,SR):
        plt.plot(1e6*np.linspace(0,len(I)/SR,len(I)),I,label = 'I')
        plt.plot(1e6*np.linspace(0,len(Q)/SR,len(I)),Q,label = 'Q')
        plt.plot(1e6*np.linspace(0,len(marker_gate)/SR,len(marker_gate)),marker_gate,label = 'gate')
        plt.plot(1e6*np.linspace(0,len(marker_acq)/SR,len(marker_acq)),marker_acq,label = 'acqisition')
        plt.plot(1e6*np.linspace(0,len(marker_dig)/SR,len(marker_dig)),marker_dig,label = 'digitiser', color = 'purple')
        plt.legend()
        plt.xlabel('Time ($\mu$s)')
        plt.ylabel('Amplitude')
        plt.show()

def silenced_echo_old(Pi,wait,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,long_wait,shot_rep,python_delay,SR = 1.28e9/(2**2),
                  inversion_wurst_pi=100e-6,phase = 0,python_avgs = 1,invert = False, pickle_waveforms = True,save= False,
                  inversion_wurst_amp = None, wait2 = None,name = "",folder = "C:\\Users\\Administrator\\Documents\\",saveraw = False):

    if inversion_wurst_amp == None: inversion_wurst_amp = wurst_amp
    if wait2==None: wait2 = 2*wait

    if invert == True:
        title2 = "With pre-inversion"
        name = name+"_inverted"

    else: title2 = "Without pre-inversion"

    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_cart = wfg.delay(150e-9)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    wurst_inversion = wfg.wurst_pulse(wurst_freq,phase,inversion_wurst_pi,amp=inversion_wurst_amp)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I = wfg.combine(d1_cart,wurst[0],d1_cart)
    wa_Q = wfg.combine(d1_cart,wurst[1],d1_cart)

    wa_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wa_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST inversion pulse

    wurst_I     = wfg.combine(d1_cart,wurst[0],d1_cart)
    wurst_Q     = wfg.combine(d1_cart,wurst[1],d1_cart)

    wurst_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wurst_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #WURST inversion pulse

    wurst_inversion_I     = wfg.combine(d1_cart,wurst_inversion[0],d1_cart)
    wurst_inversion_Q     = wfg.combine(d1_cart,wurst_inversion[1],d1_cart)

    wurst_inversion_I_neg = wfg.combine(d1_cart,-wurst_inversion[0],d1_cart)
    wurst_inversion_Q_neg = wfg.combine(d1_cart,-wurst_inversion[1],d1_cart)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_inversion_pos = create_waveform("wurst_inversion_pos",wurst_inversion_I    , wurst_inversion_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_inversion_neg = create_waveform("wurst_inversion_neg",wurst_inversion_I_neg, wurst_inversion_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # delay waveform for inter-pulse delay
    short_delay = 1e-6
    ip_delay = create_waveform("ip_delay",*[wfg.delay(short_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    r1 = wfg.delay(10e-6)
    r2 = wfg.pulse(pulse_amplitude,10e-6)
    r3 = wfg.pulse(1,10e-6)
    reset = create_waveform("reset",r2,r1,r3,r1,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)

    SE_inv_p = ['SE_inv_p',[wurst_inversion_pos,delay,In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_inv_n = ['SE_inv_n',[wurst_inversion_pos,delay,In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

    SE_p = ['SE_p',[In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()
        if invert == True: seq = SE_inv_n
        else: seq = SE_n
        if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.3") #Pickling a very long sequence can take a long time! Deactivate when not needed

        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        n_Is.append(n_I)
        n_Qs.append(n_Q)
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

        start_time = time()
        if invert == True: seq = SE_inv_p
        else: seq = SE_p
        if pickle_waveforms == True: pickle_sequence(seq,SR,address = "127.0.0.2") #Pickling a very long sequence can take a long time! Deactivate when not needed
        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        p_Is.append(p_I)
        p_Qs.append(p_Q)
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
        I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        window = 12
        t = np.multiply(downsample(timeI,window),1e9)
        I = downsample(I_demod,window)
        Q = downsample(Q_demod,window)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"_silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"_silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"_silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"_silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"_silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()

        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def silenced_echo(Pi,wait,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,long_wait,shot_rep,python_delay,SR = 1.28e9/(2**2),
                  wurst_inversion_Pi=None,phase = 0,python_avgs = 1,invert = False, pickle_input = True,pickle_output = True,
                  save= False,inversion_wurst_amp = None, wait2 = None,name = "",folder = "C:\\Users\\Administrator\\Documents\\",
                  saveraw = False,N_inv = 1,d_inv = 150e-9,window = 12,refpulse_amp = 0):
    if refpulse_amp==0: refpulse_gate=0
    else: refpulse_gate = 1
    if wurst_inversion_Pi==None: wurst_inversion_Pi = wurst_Pi
    if inversion_wurst_amp == None: inversion_wurst_amp = wurst_amp
    if wait2==None: wait2 = 2*wait

    if invert == True:
        title2 = "With pre-inversion"
        name = name+"inverted"

    else: title2 = "Without pre-inversion"

    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_cart = wfg.delay(150e-9)
    d_inv_cart = wfg.delay(d_inv)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)


    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    wurst_inversion = wfg.wurst_pulse(wurst_freq,phase,wurst_inversion_Pi,amp=inversion_wurst_amp)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I = wfg.combine(d1_cart,wurst[0],d1_cart)
    wa_Q = wfg.combine(d1_cart,wurst[1],d1_cart)

    wa_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wa_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST inversion pulse

    wurst_I     = wfg.combine(d1_cart,wurst[0],d1_cart)
    wurst_Q     = wfg.combine(d1_cart,wurst[1],d1_cart)

    wurst_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wurst_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #WURST inversion pulse

    wurst_inversion_I     = wfg.combine(d1_cart,wurst_inversion[0],d_inv_cart)
    wurst_inversion_Q     = wfg.combine(d1_cart,wurst_inversion[1],d_inv_cart)

    wurst_inversion_I_neg = wfg.combine(d1_cart,-wurst_inversion[0],d_inv_cart)
    wurst_inversion_Q_neg = wfg.combine(d1_cart,-wurst_inversion[1],d_inv_cart)

    gate_len = 150e-9-d_inv+len(wurst_inversion_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_inversion_pos = create_waveform("wurst_inversion_pos",wurst_inversion_I    , wurst_inversion_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_inversion_neg = create_waveform("wurst_inversion_neg",wurst_inversion_I_neg, wurst_inversion_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # delay waveform for inter-pulse delay
    short_delay = 1e-6
    ip_delay = create_waveform("ip_delay",*[wfg.delay(short_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    r1 = wfg.delay(10e-6)
    r2 = wfg.pulse(refpulse_amp,10e-6)
    r3 = wfg.pulse(refpulse_gate,10e-6)
    reset = create_waveform("reset",r2,r1,r3,r1,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)

    SE_inv_p = ['SE_inv_p',[wurst_inversion_pos,delay,In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_inv_n = ['SE_inv_n',[wurst_inversion_pos,delay,In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

    SE_p = ['SE_p',[In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        if invert == True: seq = SE_inv_n
        else: seq = SE_n
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        if invert == True: seq = SE_inv_p
        else: seq = SE_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def silenced_echo_heterodyne(Pi,wait,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,long_wait,shot_rep,python_delay,SR = 1.28e9/(2**2),
                  wurst_inversion_Pi=None,phase = 0,python_avgs = 1,invert = False, pickle_input = True,pickle_output = True,
                  save= False,inversion_wurst_amp = None, wait2 = None,name = "",folder = "C:\\Users\\Administrator\\Documents\\",
                  saveraw = False,N_inv = 1,d_inv = 150e-9,window = 12,refpulse_amp = 0, w=0):
    if refpulse_amp==0: refpulse_gate=0
    else: refpulse_gate = 1
    if wurst_inversion_Pi==None: wurst_inversion_Pi = wurst_Pi
    if inversion_wurst_amp == None: inversion_wurst_amp = wurst_amp
    if wait2==None: wait2 = 2*wait

    if invert == True:
        title2 = "With pre-inversion"
        name = name+"inverted"

    else: title2 = "Without pre-inversion"

    wfg.sample_rate = SR


    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_inv = wfg.heterodyne_delay(d_inv,0)
    #d1_cart = wfg.delay(150e-9)
    #d_inv_cart = wfg.delay(d_inv)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    #wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    #wurst_inversion = wfg.wurst_pulse(wurst_freq,phase,wurst_inversion_Pi,amp=inversion_wurst_amp)
    wurst = wfg.heterodyne_wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    wurst_inversion = wfg.heterodyne_wurst_pulse(wurst_freq,phase,wurst_inversion_Pi,amp=inversion_wurst_amp)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I,wa_Q     = wfg.heterodyne_combine(d1,wurst,d1,w = w,t0=0)

    wa_I_neg,wa_Q_neg = -wa_I,-wa_Q

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST inversion pulse

    wurst_I,wurst_Q     = wfg.heterodyne_combine(d1,wurst,d1,w = w,t0=0)

    wurst_I_neg,wurst_Q_neg = -wurst_I,-wurst_Q #wfg.heterodyne_combine(d1,-wurst_h,d1,w = w,t0=0)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #WURST inversion pulse

    wurst_inversion_I,wurst_inversion_Q     = wfg.heterodyne_combine(d1,wurst_inversion,d1_inv,w = w,t0=0)

    wurst_inversion_I_neg, wurst_inversion_Q_neg = -wurst_inversion_I,-wurst_inversion_Q

    #print(d_inv,len(wurst_inversion_I))

    gate_len = 150e-9-d_inv+len(wurst_inversion_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_inversion_pos = create_waveform("wurst_inversion_pos",wurst_inversion_I    , wurst_inversion_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_inversion_neg = create_waveform("wurst_inversion_neg",wurst_inversion_I_neg, wurst_inversion_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # delay waveform for inter-pulse delay
    short_delay = 1e-6
    ip_delay = create_waveform("ip_delay",*[wfg.delay(short_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    r1 = wfg.delay(10e-6)
    r2 = wfg.pulse(refpulse_amp,10e-6)
    r3 = wfg.pulse(refpulse_gate,10e-6)
    reset = create_waveform("reset",r2,r1,r3,r1,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)

    SE_inv_p = ['SE_inv_p',[wurst_inversion_pos,delay,In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_inv_n = ['SE_inv_n',[wurst_inversion_pos,delay,In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

    SE_p = ['SE_p',[In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        if invert == True: seq = SE_inv_n
        else: seq = SE_n
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        if invert == True: seq = SE_inv_p
        else: seq = SE_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def silenced_echo_arctan(Pi,wait,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,long_wait,shot_rep,python_delay,SR = 1.28e9/(2**2),
                  wurst_inversion_Pi=None,phase = 0,python_avgs = 1,invert = False, pickle_input = True,pickle_output = True,
                  save= False,inversion_wurst_amp = None, wait2 = None,name = "",folder = "C:\\Users\\Administrator\\Documents\\",
                  saveraw = False,N_inv = 1,d_inv = 150e-9,window = 12,refpulse_amp = 0, w=0, linearity = 0.2):
    if refpulse_amp==0: refpulse_gate=0
    else: refpulse_gate = 1
    if wurst_inversion_Pi==None: wurst_inversion_Pi = wurst_Pi
    if inversion_wurst_amp == None: inversion_wurst_amp = wurst_amp
    if wait2==None: wait2 = 2*wait

    if invert == True:
        title2 = "With pre-inversion"
        name = name+"inverted"

    else: title2 = "Without pre-inversion"

    wfg.sample_rate = SR


    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_inv = wfg.heterodyne_delay(d_inv,0)
    #d1_cart = wfg.delay(150e-9)
    #d_inv_cart = wfg.delay(d_inv)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    #wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    #wurst_inversion = wfg.wurst_pulse(wurst_freq,phase,wurst_inversion_Pi,amp=inversion_wurst_amp)
    wurst = wfg.heterodyne_wurst_arctan(wurst_freq,phase,wurst_Pi,amp=wurst_amp, linearity = linearity)
    wurst_inversion = wfg.heterodyne_wurst_arctan(wurst_freq,phase,wurst_inversion_Pi,amp=inversion_wurst_amp, linearity = linearity)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I,wa_Q     = wfg.heterodyne_combine(d1,wurst,d1,w = w,t0=0)

    wa_I_neg,wa_Q_neg = -wa_I,-wa_Q

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST inversion pulse

    wurst_I,wurst_Q     = wfg.heterodyne_combine(d1,wurst,d1,w = w,t0=0)

    wurst_I_neg,wurst_Q_neg = -wurst_I,-wurst_Q #wfg.heterodyne_combine(d1,-wurst_h,d1,w = w,t0=0)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #WURST inversion pulse

    wurst_inversion_I,wurst_inversion_Q     = wfg.heterodyne_combine(d1,wurst_inversion,d1_inv,w = w,t0=0)

    wurst_inversion_I_neg, wurst_inversion_Q_neg = -wurst_inversion_I,-wurst_inversion_Q

    #print(d_inv,len(wurst_inversion_I))

    gate_len = 150e-9-d_inv+len(wurst_inversion_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_inversion_pos = create_waveform("wurst_inversion_pos",wurst_inversion_I    , wurst_inversion_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_inversion_neg = create_waveform("wurst_inversion_neg",wurst_inversion_I_neg, wurst_inversion_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # delay waveform for inter-pulse delay
    short_delay = 1e-6
    ip_delay = create_waveform("ip_delay",*[wfg.delay(short_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    r1 = wfg.delay(10e-6)
    r2 = wfg.pulse(refpulse_amp,10e-6)
    r3 = wfg.pulse(refpulse_gate,10e-6)
    reset = create_waveform("reset",r2,r1,r3,r1,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)

    SE_inv_p = ['SE_inv_p',[wurst_inversion_pos,delay,In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_inv_n = ['SE_inv_n',[wurst_inversion_pos,delay,In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

    SE_p = ['SE_p',[In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        if invert == True: seq = SE_inv_n
        else: seq = SE_n
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        if invert == True: seq = SE_inv_p
        else: seq = SE_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)

def silenced_echo_sinh(Pi,wait,wurst_Pi,pulse_amplitude,wurst_amp,wurst_freq,long_wait,shot_rep,python_delay,SR = 1.28e9/(2**2),
                  wurst_inversion_Pi=None,phase = 0,python_avgs = 1,invert = False, pickle_input = True,pickle_output = True,
                  save= False,inversion_wurst_amp = None, wait2 = None,name = "",folder = "C:\\Users\\Administrator\\Documents\\",
                  saveraw = False,N_inv = 1,d_inv = 150e-9,window = 12,refpulse_amp = 0, w=0, nonlinearity = 3):
    if refpulse_amp==0: refpulse_gate=0
    else: refpulse_gate = 1
    if wurst_inversion_Pi==None: wurst_inversion_Pi = wurst_Pi
    if inversion_wurst_amp == None: inversion_wurst_amp = wurst_amp
    if wait2==None: wait2 = 2*wait

    if invert == True:
        title2 = "With pre-inversion"
        name = name+"inverted"

    else: title2 = "Without pre-inversion"

    wfg.sample_rate = SR


    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_inv = wfg.heterodyne_delay(d_inv,0)
    #d1_cart = wfg.delay(150e-9)
    #d_inv_cart = wfg.delay(d_inv)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    #wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    #wurst_inversion = wfg.wurst_pulse(wurst_freq,phase,wurst_inversion_Pi,amp=inversion_wurst_amp)
    wurst = wfg.heterodyne_wurst_sinh(wurst_freq,phase,wurst_Pi,amp=wurst_amp, nonlinearity = nonlinearity)
    wurst_inversion = wfg.heterodyne_wurst_sinh(wurst_freq,phase,wurst_inversion_Pi,amp=inversion_wurst_amp, nonlinearity = nonlinearity)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I,wa_Q     = wfg.heterodyne_combine(d1,wurst,d1,w = w,t0=0)

    wa_I_neg,wa_Q_neg = -wa_I,-wa_Q

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST inversion pulse

    wurst_I,wurst_Q     = wfg.heterodyne_combine(d1,wurst,d1,w = w,t0=0)

    wurst_I_neg,wurst_Q_neg = -wurst_I,-wurst_Q #wfg.heterodyne_combine(d1,-wurst_h,d1,w = w,t0=0)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #WURST inversion pulse

    wurst_inversion_I,wurst_inversion_Q     = wfg.heterodyne_combine(d1,wurst_inversion,d1_inv,w = w,t0=0)

    wurst_inversion_I_neg, wurst_inversion_Q_neg = -wurst_inversion_I,-wurst_inversion_Q

    #print(d_inv,len(wurst_inversion_I))

    gate_len = 150e-9-d_inv+len(wurst_inversion_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_inversion_pos = create_waveform("wurst_inversion_pos",wurst_inversion_I    , wurst_inversion_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_inversion_neg = create_waveform("wurst_inversion_neg",wurst_inversion_I_neg, wurst_inversion_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # delay waveform for inter-pulse delay
    short_delay = 1e-6
    ip_delay = create_waveform("ip_delay",*[wfg.delay(short_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    r1 = wfg.delay(10e-6)
    r2 = wfg.pulse(refpulse_amp,10e-6)
    r3 = wfg.pulse(refpulse_gate,10e-6)
    reset = create_waveform("reset",r2,r1,r3,r1,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)

    SE_inv_p = ['SE_inv_p',[wurst_inversion_pos,delay,In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_inv_n = ['SE_inv_n',[wurst_inversion_pos,delay,In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

    SE_p = ['SE_p',[In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n = ['SE_n',[In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        if invert == True: seq = SE_inv_n
        else: seq = SE_n
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        if invert == True: seq = SE_inv_p
        else: seq = SE_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)


def create_waveform(name,I,Q,marker_gate,marker_acq,marker_dig):
    waveform = [name,*wfg.length_correction(I,Q,marker_gate,marker_acq,marker_dig)]

    waveform_markers = spectr.awg.generate_MarkersArray(*waveform[3:])
    spectr.awg.ch12.load_waveform(waveform[0],
                          np.concatenate((waveform[1],waveform[2])),
                          np.concatenate((waveform_markers,waveform_markers)))
    return(waveform)

def AWG_auto_setup(SR):
    print("dummy setup AWG")

def downsample_median(data,window):
    averaged = []
    for n in range(int(len(data)/window)):
        averaged.append(np.median(data[n*window:(n+1)*window]))
    return(averaged)

def medFilter(y,x,window):
    y_filtered = downsample_median(y,window)
    x_filtered = downsample_median(x,window)
    f = interpolate.interp1d(x_filtered,y_filtered,fill_value="extrapolate")
    y_interp = f(x)
    return(x,y_interp)

def HE_3PulseRabi(shot_rep,python_delay,wait=50e-6,npts = 40, ndummy=4,HE_amp=1,phase=0, plot_waveforms = False, tag = None, folder = None):
    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0
    ###This is just a dummy sweep
    Pi =4e-6
    amps=np.linspace(1./npts,1,npts)

    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR
    pulse_amplitude = 1
    SRT = shot_rep+python_delay
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    acq_pulse_on = False
    shot = 1

    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    for i in range(ndummy):
        print('\nDummy shot %i of %i'%(shot,ndummy))
        for t_sleep in range(python_delay): sleep(1)
        ##############################################################
        #Define and load all fixed pulses and waveforms here:
        spectr.awg.ClearMemory()

        d1 = wfg.heterodyne_delay(150e-9,0)

        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)

        d2 = wfg.heterodyne_delay(wait,0)
        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)

        d3= wfg.heterodyne_delay(wait-30e-6,0)
        if acq_pulse_on ==True: acq_pulse = wfg.heterodyne_pulse(1,300e-6,phase)
        else: acq_pulse = wfg.heterodyne_delay(300e-6,phase)

        dt= wfg.pulse(single_shot,10e-6)
        dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

        ###############################################################
        #Echo detection waveform - positive piebytwo

        I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,acq_pulse,w = w,t0=0)

        gate_len = len(I_in_pos)/SR -(wait-30e-6)-300e-6
        acq_wait = gate_len+wait-20e-6#100e-9#

        marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
        marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
        marker_noacq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(0,300e-6))
        marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,230e-6),dd,dt)

        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
        noacq_markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_noacq,marker_noacq)
        ################################################################
        # delay waveform for reptime setting

        end_delay = 1e-3
        delay_w1 = wfg.delay(end_delay)

        delay_w2 = delay_w1
        delay_marker_gate = delay_w1

        marker_LED = wfg.pulse(1,end_delay)

        delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
        delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,marker_LED)
        ###############################################################

        #Set AWG parameters
        Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
        spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
        # Load waveforms
        print('loading...')

        spectr.awg.ch12.load_waveform('delay',
                                  np.concatenate((delay_w1,delay_w2)),
                                  np.concatenate((delay_markersArray,delay_markersArray)))
        spectr.awg.ch12.load_waveform('In_pos',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((markersArray,markersArray)))

        print('loaded')

        spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
        start_time = time()

        spectr.awg.ch12.init_channel('sequence_pos')
        I_raw, Q_raw, timeI, timeQ = spectr.IQ_data_raw()
        print('Time elapsed (s):',time()-start_time)

        shot+=1




    name = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H%M'))+"_"+tag
    #for t_sleep in range(10): sleep(1)

    Pi =4e-6

    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR
    pulse_amplitude = 1


    phase = 0
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    acq_pulse_on = False
    shot = 1
    start,stop = 75,120
    window = 12
    Is,Qs,mags,mag_int,I_int,Q_int = [],[],[],[],[],[]

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    for amp in amps:
        print('\nShot %i of %i\nAmp = %.2fV\n'%(shot,len(amps),amp))
        ##############################################################
        #Define and load all fixed pulses and waveforms here:
        spectr.awg.ClearMemory()
        d1 = wfg.heterodyne_delay(150e-9,0)

        pre_rotation = wfg.heterodyne_pulse(amp,Pi,phase)
        d4 = wfg.heterodyne_delay(1e-3,0)

        piebytwo = wfg.heterodyne_pulse(HE_amp/2,Pi,phase)
        piebytwo_neg = wfg.heterodyne_pulse(HE_amp/2,Pi,phase+180)

        d2 = wfg.heterodyne_delay(wait,0)
        pie = wfg.heterodyne_pulse(HE_amp,Pi,phase)
        pie_neg = wfg.heterodyne_pulse(HE_amp,Pi,phase)

        d3= wfg.heterodyne_delay(wait-30e-6,0)
        if acq_pulse_on ==True: acq_pulse = wfg.heterodyne_pulse(1,300e-6,phase)
        else: acq_pulse = wfg.heterodyne_delay(300e-6,phase)

        dt= wfg.pulse(single_shot,10e-6)
        dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

        ###############################################################
        #Echo detection waveform - positive piebytwo

        I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,acq_pulse,w = w,t0=0)
        I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,pie_neg,d3,acq_pulse,w = w,t0=0)

        gate_len = len(I_in_pos)/SR -(wait-30e-6)-300e-6
        acq_wait = gate_len+wait-20e-6#100e-9#

        marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
        marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
        marker_noacq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(0,300e-6))
        marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,230e-6),dd,dt)
        #marker_digitizer = marker_acq

        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        I_in_neg, Q_in_neg, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_neg, Q_in_neg, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
        noacq_markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_noacq,marker_noacq)

        if plot_waveforms == True:
            wfg.plot_waveforms(I_in_pos,Q_in_pos,marker_gate,marker_digitizer,SR)
            wfg.plot_waveforms(I_in_neg,Q_in_neg,marker_gate,marker_digitizer,SR)

        ################################################################
        #Pre-rotation pulse waveform

        I_pre_rot,Q_pre_rot = wfg.heterodyne_combine(d1,pre_rotation,d4,w = w,t0=0)

        gate_len = Pi + 200e-9

        pre_rot_marker_gate = wfg.combine(wfg.delay(50e-9),wfg.pulse(1,gate_len))
        pre_rot_marker_acq = wfg.delay(gate_len+50e-9)


        I_pre_rot,Q_pre_rot,pre_rot_marker_gate,pre_rot_marker_acq = wfg.length_correction(
        I_pre_rot, Q_pre_rot,pre_rot_marker_gate,pre_rot_marker_acq)

        pre_rot_markersArray = spectr.awg.generate_MarkersArray(pre_rot_marker_gate,pre_rot_marker_acq,
                                                                pre_rot_marker_acq)

        if plot_waveforms == True: wfg.plot_waveforms(I_pre_rot,Q_pre_rot,pre_rot_marker_gate,pre_rot_marker_acq,SR)

        ################################################################
        # delay waveform for reptime setting

        end_delay = 1e-3
        delay_w1 = wfg.delay(end_delay)

        delay_w2 = delay_w1
        delay_marker_gate = delay_w1

        marker_LED = wfg.pulse(1,end_delay)

        delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
        delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,marker_LED)
        ###############################################################

        #Set AWG parameters
        Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
        spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
        # Load waveforms
        print('loading...')

        spectr.awg.ch12.load_waveform('delay',
                                  np.concatenate((delay_w1,delay_w2)),
                                  np.concatenate((delay_markersArray,delay_markersArray)))
        spectr.awg.ch12.load_waveform('In_pos',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((markersArray,markersArray)))
        spectr.awg.ch12.load_waveform('In_neg',
                                  np.concatenate((I_in_neg,Q_in_neg)),
                                  np.concatenate((markersArray,markersArray)))
        spectr.awg.ch12.load_waveform('Pre_rotation',
                                  np.concatenate((I_pre_rot,Q_pre_rot)),
                                  np.concatenate((pre_rot_markersArray,pre_rot_markersArray)))
        spectr.awg.ch12.load_waveform('In_noacq',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((noacq_markersArray,noacq_markersArray)))

        print('loaded')

        spectr.awg.ch12.create_sequence('sequence_pos',['Pre_rotation','In_pos','delay'],[1,1,int(shot_rep/end_delay)])
        spectr.awg.ch12.create_sequence('sequence_neg',['Pre_rotation','In_neg','delay'],[1,1,int(shot_rep/end_delay)])
        spectr.awg.ch12.create_sequence('sequence_bsub',['Pre_rotation','In_noacq','In_pos'],[200,200,210])
        start_time = time()

        for t_sleep in range(python_delay): sleep(1)
        spectr.awg.ch12.init_channel('sequence_pos')
        #for t_sleep in range(python_delay): sleep(1)
        I_raw_p, Q_raw_p, timeI, timeQ = spectr.IQ_data_raw()
        for t_sleep in range(python_delay): sleep(1)
        spectr.awg.ch12.init_channel('sequence_neg')
        #for t_sleep in range(python_delay): sleep(1)
        I_raw_n, Q_raw_n, timeI, timeQ = spectr.IQ_data_raw()
        print('Time elapsed (s):',time()-start_time)

        #Demodulate from intermediate carrier frequency
        t_demod = np.add(wfg.time(I_raw_p[0], spectr.dig.SampleRate()),acq_wait)
        I_demod, Q_demod = wfg.signal_demod(np.mean(I_raw_p,axis=0)-np.mean(I_raw_n,axis=0),
                                            np.mean(Q_raw_p,axis = 0)-np.mean(Q_raw_n,axis = 0), t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        I = downsample(I_demod,window)
        Q = downsample(Q_demod,window)

        mag = downsample(np.mean(np.sqrt((I_raw_p-I_raw_n)**2+(Q_raw_p-Q_raw_n)**2),axis = 0),window)

        Is.append(I)
        Qs.append(Q)
        mags.append(mag)
        mag_int.append(np.mean(mag[start:stop]))
        I_int.append(np.mean(I[start:stop]))
        Q_int.append(np.mean(Q[start:stop]))



        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_%.2fHEamp_%idBm_Rabi_Is.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT,HE_amp,sgs.power.get()),Is)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_%.2fHEamp_%idBm_Rabi_Qs.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT,HE_amp,sgs.power.get()),Qs)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_%.2fHEamp_%idBm_Rabi_mags.txt' %(current_field*1e3,sgs.frequency()*1e-6,SRT,HE_amp,sgs.power.get()),mags)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_%.2fHEamp_%idBm_Rabi_amps.txt' %(current_field*1e3,sgs.frequency()*1e-6,SRT,HE_amp,sgs.power.get()),amps)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_%.2fHEamp_%idBm_Rabi_t.txt'    %(current_field*1e3,sgs.frequency()*1e-6,SRT,HE_amp,sgs.power.get()),t)

        #Plot
        #plot_pickle(amps[0:shot],mag_int,I_int)
        plot_pickle(amps[0:shot],I_int,Q_int)

        plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
        shot+=1

    plt.figure(figsize = (6,5))
    try:
        plt.clf()
        plt.plot(amps, mag_int, 'o')
        plt.xlabel(u'\u03c0/2 amplitude (V)')
        plt.ylabel('Integrated echo amplitude (V)')
        plt.tight_layout()
        plt.savefig(folder+'\\'+name+' %.3fmT%.3fMHz_mag_Rabi.pdf'%(current_field*1e3,sgs.frequency()*1e-6))
        plt.show()
    except:
        pass
    np.savetxt(folder+'\\'+name+' %.3fmT%.3fMHz_Rabi_int.txt'%(current_field*1e3,sgs.frequency()*1e-6),np.transpose([amps,I_int,Q_int,mag_int]))
    print("Done!")

    return(amps,I_int,Q_int,mag_int)

def Hahn_Echo_Rabi(shot_rep,python_delay,wait=50e-6,npts = 40, ndummy=4,refocus_amp=1,phase=0,tag = None, folder = None):
    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0
    ###This is just a dummy sweep
    Pi =4e-6
    amps=np.linspace(1./npts,1,npts)

    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR
    pulse_amplitude = 1
    SRT = shot_rep+python_delay
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    acq_pulse_on = False
    shot = 1

    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    for i in range(ndummy):
        print('\nDummy shot %i of %i'%(shot,ndummy))
        for t_sleep in range(python_delay): sleep(1)
        ##############################################################
        #Define and load all fixed pulses and waveforms here:
        spectr.awg.ClearMemory()
        d1 = wfg.heterodyne_delay(150e-9,0)
        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)

        d2 = wfg.heterodyne_delay(wait,0)
        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)

        d3= wfg.heterodyne_delay(wait-30e-6,0)
        if acq_pulse_on ==True: acq_pulse = wfg.heterodyne_pulse(1,300e-6,phase)
        else: acq_pulse = wfg.heterodyne_delay(300e-6,phase)

        dt= wfg.pulse(single_shot,10e-6)
        dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

        ###############################################################
        #Echo detection waveform - positive piebytwo

        I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,acq_pulse,w = w,t0=0)

        gate_len = len(I_in_pos)/SR -(wait-30e-6)-300e-6
        acq_wait = gate_len+wait-20e-6#100e-9#

        marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
        marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
        marker_noacq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(0,300e-6))
        marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,230e-6),dd,dt)

        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
        noacq_markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_noacq,marker_noacq)
        ################################################################
        # delay waveform for reptime setting

        end_delay = 1e-3
        delay_w1 = wfg.delay(end_delay)

        delay_w2 = delay_w1
        delay_marker_gate = delay_w1

        marker_LED = wfg.pulse(1,end_delay)

        delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
        delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,marker_LED)
        ###############################################################

        #Set AWG parameters
        Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
        spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
        # Load waveforms
        print('loading...')

        spectr.awg.ch12.load_waveform('delay',
                                  np.concatenate((delay_w1,delay_w2)),
                                  np.concatenate((delay_markersArray,delay_markersArray)))
        spectr.awg.ch12.load_waveform('In_pos',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((markersArray,markersArray)))

        print('loaded')

        spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
        start_time = time()

        spectr.awg.ch12.init_channel('sequence_pos')
        I_raw, Q_raw, timeI, timeQ = spectr.IQ_data_raw()
        print('Time elapsed (s):',time()-start_time)

        shot+=1




    name = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H%M'))+"_"+tag
    #for t_sleep in range(10): sleep(1)

    Pi =4e-6

    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR
    pulse_amplitude = 1


    phase = 0
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    acq_pulse_on = False
    shot = 1
    start,stop = 75,120
    window = 12
    Is,Qs,mags,mag_int,I_int,Q_int = [],[],[],[],[],[]

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    for amp in amps:
        print('\nShot %i of %i\nAmp = %.2fV\n'%(shot,len(amps),amp))
        ##############################################################
        #Define and load all fixed pulses and waveforms here:
        spectr.awg.ClearMemory()
        d1 = wfg.heterodyne_delay(150e-9,0)
        piebytwo = wfg.heterodyne_pulse(amp,Pi/2,phase)
        piebytwo_neg = wfg.heterodyne_pulse(amp,Pi/2,phase+180)

        d2 = wfg.heterodyne_delay(wait,0)
        pie = wfg.heterodyne_pulse(refocus_amp,Pi,phase)
        pie_neg = wfg.heterodyne_pulse(refocus_amp,Pi,phase)

        d3= wfg.heterodyne_delay(wait-30e-6,0)
        if acq_pulse_on ==True: acq_pulse = wfg.heterodyne_pulse(1,300e-6,phase)
        else: acq_pulse = wfg.heterodyne_delay(300e-6,phase)

        dt= wfg.pulse(single_shot,10e-6)
        dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

        ###############################################################
        #Echo detection waveform - positive piebytwo

        I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,acq_pulse,w = w,t0=0)
        I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,pie_neg,d3,acq_pulse,w = w,t0=0)

        gate_len = len(I_in_pos)/SR -(wait-30e-6)-300e-6
        acq_wait = gate_len+wait-20e-6#100e-9#

        marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
        marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
        marker_noacq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(0,300e-6))
        marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,230e-6),dd,dt)
        #marker_digitizer = marker_acq

        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        I_in_neg, Q_in_neg, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_neg, Q_in_neg, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
        noacq_markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_noacq,marker_noacq)
        ################################################################
        # delay waveform for reptime setting

        end_delay = 1e-3
        delay_w1 = wfg.delay(end_delay)

        delay_w2 = delay_w1
        delay_marker_gate = delay_w1

        marker_LED = wfg.pulse(1,end_delay)

        delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
        delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,marker_LED)
        ###############################################################

        #Set AWG parameters
        Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
        spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
        # Load waveforms
        print('loading...')

        spectr.awg.ch12.load_waveform('delay',
                                  np.concatenate((delay_w1,delay_w2)),
                                  np.concatenate((delay_markersArray,delay_markersArray)))
        spectr.awg.ch12.load_waveform('In_pos',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((markersArray,markersArray)))
        spectr.awg.ch12.load_waveform('In_neg',
                                  np.concatenate((I_in_neg,Q_in_neg)),
                                  np.concatenate((markersArray,markersArray)))
        spectr.awg.ch12.load_waveform('In_noacq',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((noacq_markersArray,noacq_markersArray)))

        print('loaded')

        spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
        spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','delay'],[1,int(shot_rep/end_delay)])
        spectr.awg.ch12.create_sequence('sequence_bsub',['In_noacq','In_pos'],[200,210])
        start_time = time()



        for t_sleep in range(python_delay): sleep(1)
        spectr.awg.ch12.init_channel('sequence_pos')
        #for t_sleep in range(python_delay): sleep(1)
        I_raw_p, Q_raw_p, timeI, timeQ = spectr.IQ_data_raw()
        for t_sleep in range(python_delay): sleep(1)
        spectr.awg.ch12.init_channel('sequence_neg')
        #for t_sleep in range(python_delay): sleep(1)
        I_raw_n, Q_raw_n, timeI, timeQ = spectr.IQ_data_raw()
        print('Time elapsed (s):',time()-start_time)

        #Demodulate from intermediate carrier frequency
        t_demod = np.add(wfg.time(I_raw_p[0], spectr.dig.SampleRate()),acq_wait)
        I_demod, Q_demod = wfg.signal_demod(np.mean(I_raw_p,axis=0)-np.mean(I_raw_n,axis=0),
                                            np.mean(Q_raw_p,axis = 0)-np.mean(Q_raw_n,axis = 0), t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        I = downsample(I_demod,window)
        Q = downsample(Q_demod,window)

        mag = downsample(np.mean(np.sqrt((I_raw_p-I_raw_n)**2+(Q_raw_p-Q_raw_n)**2),axis = 0),window)

        Is.append(I)
        Qs.append(Q)
        mags.append(mag)
        mag_int.append(np.mean(mag[start:stop]))
        I_int.append(np.mean(I[start:stop]))
        Q_int.append(np.mean(Q[start:stop]))


        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_Rabi_Is.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT),Is)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_Rabi_Qs.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT),Qs)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_Rabi_mags.txt' %(current_field*1e3,sgs.frequency()*1e-6,SRT),mags)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_Rabi_amps.txt' %(current_field*1e3,sgs.frequency()*1e-6,SRT),amps)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_Rabi_t.txt'    %(current_field*1e3,sgs.frequency()*1e-6,SRT),t)

        #Plot
        plot_pickle(amps[0:shot],mag_int,I_int)
        #plot_pickle(waits[0:shot],I_int,I_int)

        plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
        shot+=1

    plt.figure(figsize = (6,5))
    try:
        plt.clf()
        plt.plot(amps, mag_int, 'o')
        plt.xlabel('$\pi$/2 amp (V)')
        plt.ylabel('Echo amp (V)')
        plt.tight_layout()
        plt.savefig(folder+'\\'+name+' %.3fmT%.3fMHz_mag_Rabi.pdf'%(current_field*1e3,sgs.frequency()*1e-6))
        plt.show()
    except:
        pass
    np.savetxt(folder+'\\'+name+' %.3fmT%.3fMHz_Rabi_int.txt'%(current_field*1e3,sgs.frequency()*1e-6),np.transpose([amps,I_int,Q_int,mag_int]))
    print("Done!")
    return(I_in_pos,Q_in_pos,marker_gate,marker_digitizer,SR)

def Hahn_Echo_WURST(wurst_Pi, wurst_freq, wurst_amp, long_wait=2e-3,Pi=4e-6, SR = 1.28e9/(2**5), wait=50e-6, shot_rep=1,python_delay=9, phase=10,pulse_amplitude=1,plot=True):
    name = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H%M'))

    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    acq_pulse_on = False

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    #for t in range(40):sleep(1)
    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.delay(150e-9)

    wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)

    d2 = wfg.heterodyne_delay(150e-9,0)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d3 = wfg.heterodyne_delay(wait,0)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)

    d4= wfg.heterodyne_delay(wait-19e-6,0)

    dt = wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())
    ###############################################################
    #WURST inversion pulse

    wurst_I = wfg.combine(d1,wurst[0],d1)
    wurst_Q = wfg.combine(d1,wurst[1],d1)

    wurst_marker_acq = wfg.delay(len(wurst_I)/SR)
    if wurst_amp == 0: wurst_marker_gate = wfg.pulse(0,len(wurst_I)/SR)
    else: wurst_marker_gate = wfg.pulse(1,len(wurst_I)/SR)
    wurst_marker_digitizer = wurst_marker_acq

    wurst_I, wurst_Q, wurst_marker_gate, wurst_marker_acq,wurst_marker_digitizer = wfg.length_correction(wurst_I, wurst_Q, wurst_marker_gate,
                                                                                         wurst_marker_acq,wurst_marker_digitizer)
    wurst_markersArray = spectr.awg.generate_MarkersArray(wurst_marker_gate,wurst_marker_acq,wurst_marker_digitizer)

    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d2,piebytwo,d3,pie,w = w,t0=0)
    #I_in_pos = wfg.combine(d1,wurst[0],I_in_pos)
    #Q_in_pos = wfg.combine(d1,wurst[1],Q_in_pos)

    gate_len = len(I_in_pos)/SR
    acq_wait = gate_len+wait-20e-6

    marker_gate = wfg.combine(wfg.delay(gate_wait), wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,220e-6),dd,dt)

    I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate,
                                                                                         marker_acq,marker_digitizer)
    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    I_in_neg,Q_in_neg =  wfg.heterodyne_combine(d2,piebytwo_neg,d3,pie_neg,w = w,t0=0)
    #I_in_neg = wfg.combine(d1,wurst[0],I_in_neg)
    #Q_in_neg = wfg.combine(d1,wurst[1],Q_in_neg)

    I_in_neg, Q_in_neg, marker_gate, marker_acq,marker_digitizer = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate,
                                                                                     marker_acq,marker_digitizer)

    ###############################################################
    #delay waveform for reptime setting
    end_delay = 1e-3
    delay_w1 = wfg.delay(end_delay)
    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,delay_marker_gate)

    ###############################################################

    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms
    print('loading...')


    spectr.awg.ch12.load_waveform('wurst',
                              np.concatenate((wurst_I,wurst_Q)),
                              np.concatenate((wurst_markersArray,wurst_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))
    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))

    print('loaded')

    spectr.awg.ch12.create_sequence('sequence_pos',['wurst','delay','In_pos','delay'],[1,int(long_wait/end_delay),1,int(shot_rep/end_delay)])
    spectr.awg.ch12.create_sequence('sequence_neg',['wurst','delay','In_neg','delay'],[1,int(long_wait/end_delay),1,int(shot_rep/end_delay)])

    # Run sequences:

    start_time = time()
    for t in range(python_delay): sleep(1)
    spectr.awg.ch12.init_channel('sequence_neg')
    n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
    print('Time elapsed (s):',time()-start_time)

    #for t_sleep in range(100):sleep(1)

    start_time = time()
    for t in range(python_delay): sleep(1)
    spectr.awg.ch12.init_channel('sequence_pos')
    p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
    print('Time elapsed (s):',time()-start_time)

    PhasedI = np.subtract(p_I, n_I)
    PhasedQ = np.subtract(p_Q, n_Q)

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

    #Downsample and average
    window = 12
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)#-np.mean(I_demod[-100:])
    Q = downsample(Q_demod,window)#-np.mean(Q_demod[-100:])
    mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

    I_bgsb = np.array(I) - np.mean(I[-25:])
    Q_bgsb = np.array(Q) - np.mean(Q[-25:])
    mag_bgsb = np.sqrt(np.square(I_bgsb)+np.square(Q_bgsb))


    #Plot
    if plot == True: plot_IQmag(t,I_bgsb,Q_bgsb,mag_bgsb,title1 = 'B = %.3fmT, f = %.3fMHz'%(30,sgs.frequency()*1e-6))
    #plot_IQ(timeI*1e9,p_I,p_Q,title = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    #plot_IQ(timeI*1e9,n_I,n_Q,title = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    return I_bgsb, Q_bgsb, mag_bgsb

def Hahn_Echo_BIR(bir_theta,bir_chirp,bir_duration, bir_amp=1,long_wait=2e-3,Pi=4e-6, SR = 1.28e9/(2**5), wait=50e-6, shot_rep=1,python_delay=9, phase=10,pulse_amplitude=1,plot=True):
    name = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H%M'))

    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()


    acq_pulse_on = False


    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0


    #for t in range(40):sleep(1)
    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.delay(150e-9)

    bir = wfg.birwurst(bir_theta,bir_chirp,bir_duration,wurst_n=20,beta=np.arctan(10),amp=1,rise_and_fall=True)

    d2 = wfg.heterodyne_delay(150e-9,0)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d3 = wfg.heterodyne_delay(wait,0)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)

    d4= wfg.heterodyne_delay(wait-19e-6,0)

    dt = wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())
    ###############################################################
    #bir inversion pulse

    bir_I = wfg.combine(d1,bir[0],d1)
    bir_Q = wfg.combine(d1,bir[1],d1)

    bir_marker_acq = wfg.delay(len(bir_I)/SR)
    if bir_amp == 0: bir_marker_gate = wfg.pulse(0,len(bir_I)/SR)
    else: bir_marker_gate = wfg.pulse(1,len(bir_I)/SR)
    bir_marker_digitizer = bir_marker_acq

    bir_I, bir_Q, bir_marker_gate, bir_marker_acq,bir_marker_digitizer = wfg.length_correction(bir_I, bir_Q, bir_marker_gate,
                                                                                         bir_marker_acq,bir_marker_digitizer)
    bir_markersArray = spectr.awg.generate_MarkersArray(bir_marker_gate,bir_marker_acq,bir_marker_digitizer)

    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d2,piebytwo,d3,pie,w = w,t0=0)
    #I_in_pos = wfg.combine(d1,bir[0],I_in_pos)
    #Q_in_pos = wfg.combine(d1,bir[1],Q_in_pos)

    gate_len = len(I_in_pos)/SR
    acq_wait = gate_len+wait-20e-6

    marker_gate = wfg.combine(wfg.delay(gate_wait), wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,220e-6),dd,dt)

    I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate,
                                                                                         marker_acq,marker_digitizer)
    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    I_in_neg,Q_in_neg =  wfg.heterodyne_combine(d2,piebytwo_neg,d3,pie_neg,w = w,t0=0)
    #I_in_neg = wfg.combine(d1,bir[0],I_in_neg)
    #Q_in_neg = wfg.combine(d1,bir[1],Q_in_neg)

    I_in_neg, Q_in_neg, marker_gate, marker_acq,marker_digitizer = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate,
                                                                                     marker_acq,marker_digitizer)

    ###############################################################
    #delay waveform for reptime setting
    end_delay = 1e-3
    delay_w1 = wfg.delay(end_delay)
    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,delay_marker_gate)

    ###############################################################

    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms
    print('loading...')


    spectr.awg.ch12.load_waveform('bir',
                              np.concatenate((bir_I,bir_Q)),
                              np.concatenate((bir_markersArray,bir_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))
    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))

    print('loaded')

    spectr.awg.ch12.create_sequence('sequence_pos',['bir','delay','In_pos','delay'],[1,int(long_wait/end_delay),1,int(shot_rep/end_delay)])
    spectr.awg.ch12.create_sequence('sequence_neg',['bir','delay','In_neg','delay'],[1,int(long_wait/end_delay),1,int(shot_rep/end_delay)])

    # Run sequences:

    start_time = time()
    for t in range(python_delay): sleep(1)
    spectr.awg.ch12.init_channel('sequence_neg')
    n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
    print('Time elapsed (s):',time()-start_time)

    #for t_sleep in range(100):sleep(1)

    start_time = time()
    for t in range(python_delay): sleep(1)
    spectr.awg.ch12.init_channel('sequence_pos')
    p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
    print('Time elapsed (s):',time()-start_time)

    PhasedI = np.subtract(p_I, n_I)
    PhasedQ = np.subtract(p_Q, n_Q)

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

    #Downsample and average
    window = 12
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)#-np.mean(I_demod[-100:])
    Q = downsample(Q_demod,window)#-np.mean(Q_demod[-100:])
    mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

    I_bgsb = np.array(I) - np.mean(I[-25:])
    Q_bgsb = np.array(Q) - np.mean(Q[-25:])
    mag_bgsb = np.sqrt(np.square(I_bgsb)+np.square(Q_bgsb))


    #Plot
    if plot == True: plot_IQmag(t,I_bgsb,Q_bgsb,mag_bgsb,title1 = 'B = %.3fmT, f = %.3fMHz'%(30,sgs.frequency()*1e-6))
    #plot_IQ(timeI*1e9,p_I,p_Q,title = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    #plot_IQ(timeI*1e9,n_I,n_Q,title = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    return I_bgsb, Q_bgsb, mag_bgsb, t

def BIR_WURST_echo(bir_theta,bir_chirp,bir_duration,wurst_Pi, wurst_freq, wurst_amp, bir_amp=1,long_wait=2e-3,Pi=4e-6, SR = 1.28e9/(2**5), wait=50e-6, shot_rep=1,python_delay=9, phase=10,pulse_amplitude=1,plot=True):
    name = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H%M'))

    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()


    acq_pulse_on = False


    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0


    #for t in range(40):sleep(1)
    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.delay(150e-9)

    bir = wfg.birwurst(bir_theta,bir_chirp,bir_duration,wurst_n=20,beta=np.arctan(10),amp=1,rise_and_fall=True)

    d2 = wfg.heterodyne_delay(150e-9,0)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d3 = wfg.heterodyne_delay(wait,0)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)

    d4= wfg.heterodyne_delay(wait-19e-6,0)

    dt = wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I = wfg.combine(d1_cart,wurst[0],d1_cart)
    wa_Q = wfg.combine(d1_cart,wurst[1],d1_cart)

    wa_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wa_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,200e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,200e-6),dd,dt)

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST inversion pulse

    wurst_I     = wfg.combine(d1_cart,wurst[0],d1_cart)
    wurst_Q     = wfg.combine(d1_cart,wurst[1],d1_cart)

    wurst_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wurst_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #bir inversion pulse

    bir_I = wfg.combine(d1,bir[0],d1)
    bir_Q = wfg.combine(d1,bir[1],d1)

    bir_marker_acq = wfg.delay(len(bir_I)/SR)
    if bir_amp == 0: bir_marker_gate = wfg.pulse(0,len(bir_I)/SR)
    else: bir_marker_gate = wfg.pulse(1,len(bir_I)/SR)
    bir_marker_digitizer = bir_marker_acq

    bir_I, bir_Q, bir_marker_gate, bir_marker_acq,bir_marker_digitizer = wfg.length_correction(bir_I, bir_Q, bir_marker_gate,
                                                                                         bir_marker_acq,bir_marker_digitizer)
    bir_markersArray = spectr.awg.generate_MarkersArray(bir_marker_gate,bir_marker_acq,bir_marker_digitizer)

    ###############################################################
    #delay waveform for reptime setting
    end_delay = 1e-3
    delay_w1 = wfg.delay(end_delay)
    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,delay_marker_gate)

    ###############################################################

    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms
    print('loading...')


    spectr.awg.ch12.load_waveform('bir',
                              np.concatenate((bir_I,bir_Q)),
                              np.concatenate((bir_markersArray,bir_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))


    print('loaded')

    spectr.awg.ch12.create_sequence('sequence_pos',['bir','delay','In_pos','delay'],[1,int(long_wait/end_delay),1,int(shot_rep/end_delay)])
    spectr.awg.ch12.create_sequence('sequence_neg',['bir','delay','In_neg','delay'],[1,int(long_wait/end_delay),1,int(shot_rep/end_delay)])

    # Run sequences:

    start_time = time()
    for t in range(python_delay): sleep(1)
    spectr.awg.ch12.init_channel('sequence_neg')
    n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
    print('Time elapsed (s):',time()-start_time)

    #for t_sleep in range(100):sleep(1)

    start_time = time()
    for t in range(python_delay): sleep(1)
    spectr.awg.ch12.init_channel('sequence_pos')
    p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
    print('Time elapsed (s):',time()-start_time)

    PhasedI = np.subtract(p_I, n_I)
    PhasedQ = np.subtract(p_Q, n_Q)

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

    #Downsample and average
    window = 12
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)#-np.mean(I_demod[-100:])
    Q = downsample(Q_demod,window)#-np.mean(Q_demod[-100:])
    mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

    I_bgsb = np.array(I) - np.mean(I[-25:])
    Q_bgsb = np.array(Q) - np.mean(Q[-25:])
    mag_bgsb = np.sqrt(np.square(I_bgsb)+np.square(Q_bgsb))


    #Plot
    if plot == True: plot_IQmag(t,I_bgsb,Q_bgsb,mag_bgsb,title1 = 'B = %.3fmT, f = %.3fMHz'%(30,sgs.frequency()*1e-6))
    #plot_IQ(timeI*1e9,p_I,p_Q,title = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    #plot_IQ(timeI*1e9,n_I,n_Q,title = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    return I_bgsb, Q_bgsb, mag_bgsb, t

def Wurst_refocus(wurst_Pi, wurst_freq, wurst_amp,shot_rep=1,python_delay = 9,wait = 50e-6,pulse_amplitude = 1,SR = 1.28e9/(2**5),Pi=4e-6,phase = 10, plot=True, save=False,plot_waveforms=False):
    name = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H%M'))

    wfg.sample_rate = SR
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()
    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0


    acq_pulse_on = False



    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)

    wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)

    d2 = wfg.heterodyne_delay(150e-9,0)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d3 = wfg.delay(wait)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)

    d4= wfg.delay(wait-19e-6)

    dt = wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())
    ###############################################################
    #WURST inversion pulse

    wurst_I = wfg.combine(d3,wurst[0],d3,d3,wurst[0])
    wurst_Q = wfg.combine(d3,wurst[1],d3,d3,wurst[1])

    wurst_I_neg = wfg.combine(d3,-wurst[0],d3,d3,-wurst[0])
    wurst_Q_neg = wfg.combine(d3,-wurst[1],d3,d3,-wurst[1])

    gate_len = len(wurst_I)/SR
    acq_wait = gate_len+wait-20e-6

    wurst_marker_gate = wfg.combine(wfg.delay(wait), wfg.pulse(1,len(wurst[0])/SR),wfg.delay(2*wait-200e-9),wfg.pulse(1,len(wurst[0])/SR))
    wurst_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    wurst_marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,220e-6),dd,dt)

    wurst_I, wurst_Q, wurst_marker_gate, wurst_marker_acq,wurst_marker_digitizer = wfg.length_correction(wurst_I, wurst_Q, wurst_marker_gate,
                                                                                                         wurst_marker_acq,wurst_marker_digitizer)
    wurst_markersArray = spectr.awg.generate_MarkersArray(wurst_marker_gate,wurst_marker_acq,wurst_marker_digitizer)

    wurst_I_neg, wurst_Q_neg, wurst_marker_gate, wurst_marker_acq,wurst_marker_digitizer = wfg.length_correction(wurst_I_neg, wurst_Q_neg, wurst_marker_gate,
                                                                                                         wurst_marker_acq,wurst_marker_digitizer)
    wurst_neg_markersArray = spectr.awg.generate_MarkersArray(wurst_marker_gate,wurst_marker_acq,wurst_marker_digitizer)
    if plot_waveforms == True: wfg.plot_waveforms(Iwurst_I, wurst_Q,wurst_marker_gate, wurst_marker_acq,SR)
    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d1,w=w,t0=0)
    #I_in_pos = wfg.combine(d1,wurst[0],I_in_pos)
    #Q_in_pos = wfg.combine(d1,wurst[1],Q_in_pos)

    gate_len = len(I_in_pos)/SR
    acq_wait = gate_len+wait-20e-6

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq =  wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    I_in_neg,Q_in_neg =  wfg.heterodyne_combine(d1,piebytwo_neg,d1,w=w,t0=0)
    #I_in_neg = wfg.combine(d1,wurst[0],I_in_neg)
    #Q_in_neg = wfg.combine(d1,wurst[1],Q_in_neg)

    I_in_neg, Q_in_neg, marker_gate, marker_acq,marker_digitizer = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    ###############################################################
    #delay waveform for reptime setting
    end_delay = 1e-3
    delay_w1 = wfg.delay(end_delay)
    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,delay_marker_gate)

    ###############################################################

    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms
    print('loading...')


    spectr.awg.ch12.load_waveform('wurst',
                              np.concatenate((wurst_I,wurst_Q)),
                              np.concatenate((wurst_markersArray,wurst_markersArray)))
    spectr.awg.ch12.load_waveform('wurst_neg',
                              np.concatenate((wurst_I_neg,wurst_Q_neg)),
                              np.concatenate((wurst_neg_markersArray,wurst_neg_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))
    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))



    print('loaded')

    spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','wurst',    'delay'],[1,1,int(shot_rep/end_delay)])
    spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','wurst_neg','delay'],[1,1,int(shot_rep/end_delay)])

    # Run sequences:

    start_time = time()
    for t_sleep in range(python_delay):sleep(1)
    spectr.awg.ch12.init_channel('sequence_neg')
    n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
    print('Time elapsed (s):',time()-start_time)

    start_time = time()
    for t_sleep in range(python_delay):sleep(1)
    spectr.awg.ch12.init_channel('sequence_pos')
    p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
    print('Time elapsed (s):',time()-start_time)

    PhasedI = np.subtract(p_I, n_I)
    PhasedQ = np.subtract(p_Q, n_Q)

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

    #Downsample and average
    window = 12
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)#-np.mean(I_demod[-100:])
    Q = downsample(Q_demod,window)#-np.mean(Q_demod[-100:])
    mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

    I_bgsb = np.array(I) - np.mean(I[-25:])
    Q_bgsb = np.array(Q) - np.mean(Q[-25:])
    mag_bgsb = np.sqrt(np.square(I_bgsb)+np.square(Q_bgsb))


    #Plot
    if plot == True: plot_IQmag(t,I_bgsb,Q_bgsb,mag_bgsb,title1 = 'B = %.3fmT, f = %.3fMHz'%(30,sgs.frequency()*1e-6))
    #plot_IQ(timeI*1e9,p_I,p_Q,title = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    #plot_IQ(timeI*1e9,n_I,n_Q,title = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    if save == True:
        np.savetxt(folder+"\\"+name+"wait_%.1f_echo_data_Is.txt"%wait,I_bgsb)
        np.savetxt(folder+"\\"+name+"wait_%.1f_echo_data_Qs.txt"%wait,Q_bgsb)
        np.savetxt(folder+"\\"+name+"wait_%.1f_echo_data_mags.txt"%wait,mag_bgsb)
        np.savetxt(folder+"\\"+name+"wait_%.1f_echo_data_ts.txt"%wait,t)
    return I_bgsb, Q_bgsb, mag_bgsb, t

def silenced_echo_CPMG(Pi,wait,wurst_Pi,short_wurst_Pi,pulse_amplitude,wurst_amp,wurst_amp_short,wurst_freq,wurst_freq_short,long_wait,shot_rep, python_delay,N_refocusings=1,phase = 0,python_avgs = 1,invert = False, pickle_wavforms = True,save= False, acq_on_excitation=False):
    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(150e-9,0)
    d1_cart = wfg.delay(150e-9)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)
    d2_cart = wfg.delay(wait)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    wurst = wfg.wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    short_wurst = wfg.wurst_pulse(wurst_freq_short,phase,short_wurst_Pi,amp=wurst_amp_short)

    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I = wfg.combine(d1_cart,wurst[0],d1_cart)
    wa_Q = wfg.combine(d1_cart,wurst[1],d1_cart)

    wa_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wa_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,200e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,200e-6),dd,dt)

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST inversion pulse

    wurst_I     = wfg.combine(d1_cart,wurst[0],d1_cart)
    wurst_Q     = wfg.combine(d1_cart,wurst[1],d1_cart)

    wurst_I_neg = wfg.combine(d1_cart,-wurst[0],d1_cart)
    wurst_Q_neg = wfg.combine(d1_cart,-wurst[1],d1_cart)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #WURST DD refocusing pulse

    short_wurst_I     = wfg.combine(d1_cart,short_wurst[0],d2_cart,d2_cart,short_wurst[0],d2_cart,d2_cart)
    short_wurst_Q     = wfg.combine(d1_cart,short_wurst[1],d2_cart,d2_cart,short_wurst[1],d2_cart,d2_cart)

    short_wurst_I_neg = wfg.combine(d1_cart,-short_wurst[0],d2_cart,d2_cart,short_wurst[0],d2_cart,d2_cart)
    short_wurst_Q_neg = wfg.combine(d1_cart,-short_wurst[1],d2_cart,d2_cart,short_wurst[1],d2_cart,d2_cart)

    gate_len = len(short_wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.combine(wfg.pulse(1,short_wurst_Pi + 300e-9), wfg.delay(2*wait-300e-9), wfg.pulse(1,short_wurst_Pi + 300e-9))
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    refocus_wurst_pos = create_waveform("refocus_wurst_pos",short_wurst_I    , short_wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    refocus_wurst_neg = create_waveform("refocus_wurst_neg",short_wurst_I_neg, short_wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)
    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR
    acq_wait = 0

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.combine(wfg.pulse(1,gate_len))

    In_pos_a = create_waveform("In_pos_a",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg_a = create_waveform("In_neg_a",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)
    ###############################################################
    #single shot triggers
    I_trig,Q_trig = d1_cart, d1_cart

    gate_len = len(I_in_pos)/SR
    acq_wait = 0

    marker_gate = wfg.delay(gate_len)
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.combine(dd,dt)

    bonus_trigger = create_waveform("bonus_trigger",I_trig,Q_trig, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)
    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # delay waveform for inter-pulse delay
    short_delay = 1e-6
    ip_delay = create_waveform("ip_delay",*[wfg.delay(short_delay)]*5)
    ###############################################################
    AWG_auto_setup(SR)
    if acq_on_excitation == True:
        SE_inv_p = ['SE_inv_p',[wurst_pos,delay,In_pos_a,wurst_pos,ip_delay,refocus_wurst_pos,wurst_pos,bonus_trigger,delay],
                        [1,int(long_wait/end_delay),1,1,int(wait*2/short_delay),N_refocusings,1,1,int(shot_rep/end_delay)]]
        spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

        SE_inv_n = ['SE_inv_n',[wurst_neg,delay,In_neg_a,wurst_pos,ip_delay,refocus_wurst_pos,wurst_pos,bonus_trigger,delay],
                        [1,int(long_wait/end_delay),1,1,int(wait*2/short_delay),N_refocusings,1,1,int(shot_rep/end_delay)]]
        spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

        SE_p = ['SE_p',[In_pos_a,wurst_pos,ip_delay,refocus_wurst_pos,wurst_pos,bonus_trigger,delay],
                        [1,1,int(wait*2/short_delay),N_refocusings,1,1,int(shot_rep/end_delay)]]
        spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

        SE_n = ['SE_n',[In_neg_a,wurst_pos,ip_delay,refocus_wurst_pos,wurst_pos,bonus_trigger,delay],
                        [1,1,int(wait*2/short_delay),N_refocusings,1,1,int(shot_rep/end_delay)]]
        spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])
    else:
        SE_inv_p = ['SE_inv_p',[wurst_pos,delay,In_pos,wurst_pos,ip_delay,refocus_wurst_pos,wa_pos,delay],
                        [1,int(long_wait/end_delay),1,1,int(wait*2/short_delay),N_refocusings,1,int(shot_rep/end_delay)]]
        spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

        SE_inv_n = ['SE_inv_n',[wurst_neg,delay,In_neg,wurst_pos,ip_delay,refocus_wurst_pos,wa_pos,delay],
                        [1,int(long_wait/end_delay),1,1,int(wait*2/short_delay),N_refocusings,1,int(shot_rep/end_delay)]]
        spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

        SE_p = ['SE_p',[In_pos,wurst_pos,ip_delay,refocus_wurst_pos,wa_pos,delay],
                        [1,1,int(wait*2/short_delay),N_refocusings,1,int(shot_rep/end_delay)]]
        spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

        SE_n = ['SE_n',[In_neg,wurst_pos,ip_delay,refocus_wurst_pos,wa_pos,delay],
                        [1,1,int(wait*2/short_delay),N_refocusings,1,int(shot_rep/end_delay)]]
        spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')
    ###############################################################

    # Run sequences: positive and negative for phase cycling
    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()
        if invert == True: seq = SE_inv_n
        else: seq = SE_n
        if pickle_wavforms == True: pickle_sequence(seq,address = "127.0.0.3") #Pickling a very long sequence can take a long time! Deactivate when not needed

        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        n_Is.append(n_I)
        n_Qs.append(n_Q)
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)
        if save==True:
                np.savetxt(folder+"\\"+name+"silencedEcho_n_I.txt",n_I)
                np.savetxt(folder+"\\"+name+"silencedEcho_n_Q.txt",n_Q)
        print('Average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    for n in range(python_avgs):
        start_time = time()
        if invert == True: seq = SE_inv_p
        else: seq = SE_p
        if pickle_wavforms == True: pickle_sequence(seq,address = "127.0.0.2") #Pickling a very long sequence can take a long time! Deactivate when not needed
        while (time()-start_time)<(python_delay-0.05): sleep(0.1)
        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        p_Is.append(p_I)
        p_Qs.append(p_Q)
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)
        if save==True:
                np.savetxt(folder+"\\"+name+"silencedEcho_p_I.txt",p_I)
                np.savetxt(folder+"\\"+name+"silencedEcho_p_Q.txt",p_Q)
        print('Average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    PhasedI = np.subtract(p_I, n_I)
    PhasedQ = np.subtract(p_Q, n_Q)

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

    #Downsample and average
    window = 12
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)#-np.mean(I_demod[-100:])
    Q = downsample(Q_demod,window)#-np.mean(Q_demod[-100:])
    mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

    #Plot
    s = 5
    t1,I1,Q1,mag1 = t[s:],I[s:],Q[s:],mag[s:]
    plot_pickle(t1/1e9,I1,Q1)
    if invert == True: title2 = "With pre-inversion"
    else: title2 = "Without pre-inversion"
    plot_IQmag(t1,I1,Q1,mag1,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    return(t1,I1,Q1,mag1)

def run_T2_steadyState(shot_rep,python_delay,I_bgs,Q_bgs,npts = 40,n_dummy=4, tag = None, folder = None,bsub=False, medwindow = 10,phase = 0,Pi=4e-6):


    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0
    ###This is just a dummy sweep
    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR
    pulse_amplitude = 1
    SRT = shot_rep+python_delay

    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    waits = 40e-6*np.round(np.logspace(0,0.1,n_dummy),5)
    # waits = 40e-6*np.round(np.logspace(0,2.4,80),2)

    acq_pulse_on = False
    shot = 1

    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    for wait in waits:
        print('\nDummy shot %i of %i\nTau = %.ius\n'%(shot,len(waits),1e6*wait))
        for t_sleep in range(python_delay): sleep(1)
        ##############################################################
        #Define and load all fixed pulses and waveforms here:
        spectr.awg.ClearMemory()
        d1 = wfg.heterodyne_delay(150e-9,0)
        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)

        d2 = wfg.heterodyne_delay(wait,0)
        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)

        d3= wfg.heterodyne_delay(wait-30e-6,0)
        if acq_pulse_on ==True: acq_pulse = wfg.heterodyne_pulse(1,300e-6,phase)
        else: acq_pulse = wfg.heterodyne_delay(300e-6,phase)

        dt= wfg.pulse(single_shot,10e-6)
        dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

        ###############################################################
        #Echo detection waveform - positive piebytwo

        I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,acq_pulse,w = w,t0=0)

        gate_len = len(I_in_pos)/SR -(wait-30e-6)-300e-6
        acq_wait = gate_len+wait-20e-6#100e-9#

        marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
        marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
        marker_noacq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(0,300e-6))
        marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,230e-6),dd,dt)

        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
        noacq_markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_noacq,marker_noacq)
        ################################################################
        # delay waveform for reptime setting

        end_delay = 1e-3
        delay_w1 = wfg.delay(end_delay)

        delay_w2 = delay_w1
        delay_marker_gate = delay_w1

        marker_LED = wfg.pulse(1,end_delay)

        delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
        delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,marker_LED)
        ###############################################################

        #Set AWG parameters
        Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
        spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
        # Load waveforms
        print('loading...')

        spectr.awg.ch12.load_waveform('delay',
                                  np.concatenate((delay_w1,delay_w2)),
                                  np.concatenate((delay_markersArray,delay_markersArray)))
        spectr.awg.ch12.load_waveform('In_pos',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((markersArray,markersArray)))

        print('loaded')

        spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
        start_time = time()

        spectr.awg.ch12.init_channel('sequence_pos')
        I_raw, Q_raw, timeI, timeQ = spectr.IQ_data_raw()
        print('Time elapsed (s):',time()-start_time)

        shot+=1




    name = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H%M'))+"_"+tag
    #for t_sleep in range(10): sleep(1)

    #Pi =4e-6

    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR
    pulse_amplitude = 1



    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    waits = 40e-6*np.round(np.logspace(0,1.4,npts),2)
    # waits = 40e-6*np.round(np.logspace(0,.1,2),2)

    acq_pulse_on = False
    shot = 1
    start,stop = 75,120
    window = 12
    Is,Qs,mags,mag_int,I_int,Q_int = [],[],[],[],[],[]

    if bsub==True:
        print("acquiring background traces...")
        I_bgs,Q_bgs,python_delay,single_shot =[],[],0,0

    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    for wait in waits:
        print('\nShot %i of %i\nTau = %.ius\n'%(shot,len(waits),1e6*wait))
        for t_sleep in range(python_delay): sleep(1)
        ##############################################################
        #Define and load all fixed pulses and waveforms here:
        spectr.awg.ClearMemory()
        d1 = wfg.heterodyne_delay(150e-9,0)
        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)

        d2 = wfg.heterodyne_delay(wait,0)
        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)

        d3= wfg.heterodyne_delay(wait-30e-6,0)
        if acq_pulse_on ==True: acq_pulse = wfg.heterodyne_pulse(1,300e-6,phase)
        else: acq_pulse = wfg.heterodyne_delay(300e-6,phase)

        dt= wfg.pulse(single_shot,10e-6)
        dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

        ###############################################################
        #Echo detection waveform - positive piebytwo

        I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,acq_pulse,w = w,t0=0)

        gate_len = len(I_in_pos)/SR -(wait-30e-6)-300e-6
        acq_wait = gate_len+wait-20e-6#100e-9#

        marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
        marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
        marker_noacq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(0,300e-6))
        marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,230e-6),dd,dt)
        #marker_digitizer = marker_acq

        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
        noacq_markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_noacq,marker_noacq)
        ################################################################
        # delay waveform for reptime setting

        end_delay = 1e-3
        delay_w1 = wfg.delay(end_delay)

        delay_w2 = delay_w1
        delay_marker_gate = delay_w1

        marker_LED = wfg.pulse(1,end_delay)

        delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
        delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,marker_LED)
        ###############################################################

        #Set AWG parameters
        Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
        spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
        # Load waveforms
        print('loading...')

        spectr.awg.ch12.load_waveform('delay',
                                  np.concatenate((delay_w1,delay_w2)),
                                  np.concatenate((delay_markersArray,delay_markersArray)))
        spectr.awg.ch12.load_waveform('In_pos',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((markersArray,markersArray)))
        spectr.awg.ch12.load_waveform('In_noacq',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((noacq_markersArray,noacq_markersArray)))

        print('loaded')

        spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
        spectr.awg.ch12.create_sequence('sequence_bsub',['In_noacq','In_pos'],[200,210])
        start_time = time()

        if bsub ==True:
            nacq = spectr.NumberOfAcquisitions()
            spectr.NumberOfAcquisitions(400)
            spectr.awg.ch12.init_channel('sequence_bsub')
            I_bg, Q_bg, timeI, timeQ = spectr.IQ_data_averaged()

            plot_IQ(np.multiply(downsample(timeI,window),1e9),downsample(I_bg,window),downsample(Q_bg,window),show = False)
            t_echo,I_bg = medFilter(I_bg,timeI,medwindow)
            t_echo,Q_bg = medFilter(Q_bg,timeI,medwindow)
            I_bgs.append(I_bg)
            Q_bgs.append(Q_bg)

            plot_IQ(np.multiply(downsample(timeI,window),1e9),downsample(I_bg,window),downsample(Q_bg,window))
            spectr.NumberOfAcquisitions(nacq)
            #for t_sleep in range(int(python_delay)): sleep(1)


            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_I_bgs.txt'%(current_field*1e3,sgs.frequency()*1e-6,SRT),I_bgs)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_Q_bgs.txt'%(current_field*1e3,sgs.frequency()*1e-6,SRT),Q_bgs)
        else:
            spectr.awg.ch12.init_channel('sequence_pos')
            #for t_sleep in range(python_delay): sleep(1)
            I_raw, Q_raw, timeI, timeQ = spectr.IQ_data_raw()
            print('Time elapsed (s):',time()-start_time)

            #Demodulate from intermediate carrier frequency
            t_demod = np.add(wfg.time(I_raw[0], spectr.dig.SampleRate()),acq_wait)
            I_demod, Q_demod = wfg.signal_demod(np.mean(I_raw,axis=0)-I_bgs[shot-1], np.mean(Q_raw,axis = 0)-Q_bgs[shot-1],
                                                t_demod, w)

            #Downsample and average
            t = np.multiply(downsample(timeI,window),1e9)
            I = downsample(I_demod,window)
            Q = downsample(Q_demod,window)

            mag = downsample(np.mean(np.sqrt((I_raw-I_bgs[shot-1])**2+(Q_raw-Q_bgs[shot-1])**2),axis = 0),window)

            Is.append(I)
            Qs.append(Q)
            mags.append(mag)
            mag_int.append(np.mean(mag[start:stop]))
            I_int.append(np.mean(I[start:stop]))
            Q_int.append(np.mean(Q[start:stop]))


            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_Is.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT),Is)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_Qs.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT),Qs)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_mags.txt' %(current_field*1e3,sgs.frequency()*1e-6,SRT),mags)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_waits.txt'%(current_field*1e3,sgs.frequency()*1e-6,SRT),waits)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_t.txt'    %(current_field*1e3,sgs.frequency()*1e-6,SRT),t)

            #Plot
            plot_pickle(waits[0:shot],mag_int,I_int)
            #plot_pickle(waits[0:shot],I_int,I_int)

            plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
        shot+=1

    plt.figure(figsize = (6,5))
    try:
        T2_stretchedfit(mag_int,waits,noise_floor = np.mean(mag[-50:]),title = 'Magnitude, B = %.3fmT, f = %.3fMHz'%(current_field*1e3,sgs.frequency()*1e-6),
               save = False,filename = folder+'\\'+name+' %.3fmT%.3fMHz_T2_mag.pdf' %(current_field*1e3,sgs.frequency()*1e-6),show =False)
        #plt.xlim(0,520)
        #plt.ylim(0,None)
        plt.tight_layout()
        plt.savefig(folder+'\\'+name+' %.3fmT%.3fMHz_mag_T2.pdf'%(current_field*1e3,sgs.frequency()*1e-6))
    except:
        pass
    if bsub == False: np.savetxt(folder+'\\'+name+' %.3fmT%.3fMHz_T2_int.txt'%(current_field*1e3,sgs.frequency()*1e-6),np.transpose([waits,I_int,Q_int,mag_int]))
    print("Done!")
    if bsub==True: return I_bgs, Q_bgs
    else: return Is,Qs,mags,waits,t

def run_T2_phased_cycled(shot_rep,python_delay,npts=40,n_dummy=4,tag = None,folder = None):
    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0
    ###This is just a dummy sweep
    Pi =10e-6
    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR
    pulse_amplitude = 1
    SRT = shot_rep+python_delay
    phase = 0
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()
    start,stop = 75,120

    waits = 40e-6*np.round(np.logspace(0,0.1,n_dummy),5)
    # waits = 40e-6*np.round(np.logspace(0,2.4,80),2)

    acq_pulse_on = False
    shot = 1

    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    Is,Qs,mags,mag_int,I_int,Q_int = [],[],[],[],[],[]

    for wait in waits:
        #Define and load all fixed pulses and waveforms here:
        spectr.awg.ClearMemory()
        d1 = wfg.heterodyne_delay(150e-9,0)

        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
        piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

        d2 = wfg.heterodyne_delay(wait,0)

        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
        pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+0) # leave identical to pie unless you want to phase cycle the pi pulse!

        d3= wfg.heterodyne_delay(wait-25e-6,0)
        if acq_pulse_on ==True: acq_pulse = wfg.heterodyne_pulse(1,300e-6,phase)
        else: acq_pulse = wfg.heterodyne_delay(300e-6,phase)

        dt= wfg.pulse(single_shot,10e-6)
        dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

        ###############################################################
        #Echo detection waveform - positive piebytwo

        I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,acq_pulse,w = w,t0=0)

        #gate_len = len(I_in_pos)/SR + 100e-9
        gate_len = len(I_in_pos)/SR -(wait-25e-6)-300e-6
        acq_wait = gate_len+wait-20e-6#

        marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
        marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
        marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,220e-6),dd,dt)

        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate,
                                                                                             marker_acq,marker_digitizer)
        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

        #Echo detection waveform - negative piebytwo

        I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,pie_neg,d3,acq_pulse,w = w,t0=0)
        I_in_neg, Q_in_neg, marker_gate, marker_acq,marker_digitizer = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate,
                                                                                             marker_acq,marker_digitizer)
        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
        ################################################################
        # delay waveform for reptime setting

        end_delay = 1e-3
        delay_w1 = wfg.delay(end_delay)

        delay_w2 = delay_w1
        delay_marker_gate = delay_w1

        marker_LED = wfg.pulse(1,end_delay)

        delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
        delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,marker_LED)
        ###############################################################

        #Set AWG parameters
        Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
        spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
        # Load waveforms
        print('loading...')

        spectr.awg.ch12.load_waveform('delay',
                                  np.concatenate((delay_w1,delay_w2)),
                                  np.concatenate((delay_markersArray,delay_markersArray)))
        spectr.awg.ch12.load_waveform('In_pos',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((markersArray,markersArray)))
        spectr.awg.ch12.load_waveform('In_neg',
                                  np.concatenate((I_in_neg,Q_in_neg)),
                                  np.concatenate((markersArray,markersArray)))
        print('loaded')

        spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
        spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','delay'],[1,int(shot_rep/end_delay)])

        # Run sequences: positive and negative for phase cycling
        start_time = time()
        spectr.awg.ch12.init_channel('sequence_neg')
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        print('Time elapsed (s):',time()-start_time)

        #for t_sleep in range(100):sleep(1)

        start_time = time()

        spectr.awg.ch12.init_channel('sequence_pos')
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        print('Time elapsed (s):',time()-start_time)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
        I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        window = 12
        t = np.multiply(downsample(timeI,window),1e9)
        I = downsample(I_demod,window)#-np.mean(I_demod[-100:])
        Q = downsample(Q_demod,window)#-np.mean(Q_demod[-100:])
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        s = 5
        t1,I1,Q1,mag1 = t[s:],I[s:],Q[s:],mag[s:]
        plot_pickle(t1,I1,Q1)
        plot_IQmag(t1,I1,Q1,mag1,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))

        Is.append(I)
        Qs.append(Q)
        mags.append(mag)
        mag_int.append(np.mean(mag[start:stop]))
        I_int.append(np.mean(I[start:stop]))
        Q_int.append(np.mean(Q[start:stop]))

        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_Is.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT),Is)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_Qs.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT),Qs)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_mags.txt' %(current_field*1e3,sgs.frequency()*1e-6,SRT),mags)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_waits.txt'%(current_field*1e3,sgs.frequency()*1e-6,SRT),waits)
        np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_t.txt'    %(current_field*1e3,sgs.frequency()*1e-6,SRT),t)

        shot+=1
    plt.figure(figsize = (6,5))
    try:
        T2_stretchedfit(mag_int,waits,noise_floor = np.mean(mag[-50:]),title = 'Magnitude, B = %.3fmT, f = %.3fMHz'%(current_field*1e3,sgs.frequency()*1e-6),
               save = False,filename = folder+'\\'+name+' %.3fmT%.3fMHz_T2_mag.pdf' %(current_field*1e3,sgs.frequency()*1e-6),show =False)
        #plt.xlim(0,520)
        #plt.ylim(0,None)
        plt.tight_layout()
        plt.savefig(folder+'\\'+name+' %.3fmT%.3fMHz_mag_T2.pdf'%(current_field*1e3,sgs.frequency()*1e-6))
    except:
        pass

    return 0

def run_T2_steadyState_weak_refocus(shot_rep,python_delay,refocus_amp,I_bgs,Q_bgs,npts = 40, tag = None, folder = None):
    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0
    ###This is just a dummy sweep
    Pi =4e-6

    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR
    pulse_amplitude = 1

    SRT = shot_rep+python_delay
    phase = 0
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    waits = 40e-6*np.round(np.logspace(0,0.1,4),2)
    # waits = 40e-6*np.round(np.logspace(0,2.4,80),2)

    acq_pulse_on = False
    shot = 1

    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    for wait in waits:
        print('\nDummy shot %i of %i\nTau = %.ius\n'%(shot,len(waits),1e6*wait))
        for t_sleep in range(python_delay): sleep(1)
        ##############################################################
        #Define and load all fixed pulses and waveforms here:
        spectr.awg.ClearMemory()
        d1 = wfg.heterodyne_delay(150e-9,0)
        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)

        d2 = wfg.heterodyne_delay(wait,0)
        pie = wfg.heterodyne_pulse(refocus_amp,Pi,phase)

        d3= wfg.heterodyne_delay(wait-30e-6,0)
        if acq_pulse_on ==True: acq_pulse = wfg.heterodyne_pulse(1,300e-6,phase)
        else: acq_pulse = wfg.heterodyne_delay(300e-6,phase)

        dt= wfg.pulse(single_shot,10e-6)
        dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

        ###############################################################
        #Echo detection waveform - positive piebytwo

        I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,acq_pulse,w = w,t0=0)

        gate_len = len(I_in_pos)/SR -(wait-30e-6)-300e-6
        acq_wait = gate_len+wait-20e-6#100e-9#

        marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
        marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
        marker_noacq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(0,300e-6))
        marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,230e-6),dd,dt)

        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
        noacq_markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_noacq,marker_noacq)
        ################################################################
        # delay waveform for reptime setting

        end_delay = 1e-3
        delay_w1 = wfg.delay(end_delay)

        delay_w2 = delay_w1
        delay_marker_gate = delay_w1

        marker_LED = wfg.pulse(1,end_delay)

        delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
        delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,marker_LED)
        ###############################################################

        #Set AWG parameters
        Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
        spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
        # Load waveforms
        print('loading...')

        spectr.awg.ch12.load_waveform('delay',
                                  np.concatenate((delay_w1,delay_w2)),
                                  np.concatenate((delay_markersArray,delay_markersArray)))
        spectr.awg.ch12.load_waveform('In_pos',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((markersArray,markersArray)))

        print('loaded')

        spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
        start_time = time()

        spectr.awg.ch12.init_channel('sequence_pos')
        I_raw, Q_raw, timeI, timeQ = spectr.IQ_data_raw()
        print('Time elapsed (s):',time()-start_time)

        shot+=1




    name = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H%M'))+"_"+tag
    #for t_sleep in range(10): sleep(1)

    Pi =4e-6

    SR = 1.28e9/(2**2)
    wfg.sample_rate = SR
    pulse_amplitude = 1


    phase = 0
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    waits = 40e-6*np.round(np.logspace(0,1.4,npts),2)
    # waits = 40e-6*np.round(np.logspace(0,.1,2),2)

    acq_pulse_on = False
    shot = 1
    start,stop = 75,120
    window = 12
    Is,Qs,mags,mag_int,I_int,Q_int = [],[],[],[],[],[]

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    #bsub = True
    bsub = False
    if bsub==True:
        print("acquiring background traces...")
        I_bgs,Q_bgs,python_delay,single_shot =[],[],0,0

    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    for wait in waits:
        print('\nShot %i of %i\nTau = %.ius\n'%(shot,len(waits),1e6*wait))
        for t_sleep in range(python_delay): sleep(1)
        ##############################################################
        #Define and load all fixed pulses and waveforms here:
        spectr.awg.ClearMemory()
        d1 = wfg.heterodyne_delay(150e-9,0)
        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)

        d2 = wfg.heterodyne_delay(wait,0)
        pie = wfg.heterodyne_pulse(refocus_amp,Pi,phase)

        d3= wfg.heterodyne_delay(wait-30e-6,0)
        if acq_pulse_on ==True: acq_pulse = wfg.heterodyne_pulse(1,300e-6,phase)
        else: acq_pulse = wfg.heterodyne_delay(300e-6,phase)

        dt= wfg.pulse(single_shot,10e-6)
        dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

        ###############################################################
        #Echo detection waveform - positive piebytwo

        I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,acq_pulse,w = w,t0=0)

        gate_len = len(I_in_pos)/SR -(wait-30e-6)-300e-6
        acq_wait = gate_len+wait-20e-6#100e-9#

        marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9),wfg.delay(wait-200e-9),wfg.pulse(1,Pi+200e-9))
        marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
        marker_noacq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(0,300e-6))
        marker_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,230e-6),dd,dt)
        #marker_digitizer = marker_acq

        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq = wfg.length_correction(
        I_in_pos, Q_in_pos, marker_gate, marker_acq,marker_digitizer, marker_noacq)

        markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
        noacq_markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_noacq,marker_noacq)
        ################################################################
        # delay waveform for reptime setting

        end_delay = 1e-3
        delay_w1 = wfg.delay(end_delay)

        delay_w2 = delay_w1
        delay_marker_gate = delay_w1

        marker_LED = wfg.pulse(1,end_delay)

        delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
        delay_markersArray = spectr.awg.generate_MarkersArray(delay_w1,delay_w2,delay_marker_gate,marker_LED)
        ###############################################################

        #Set AWG parameters
        Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
        spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
        # Load waveforms
        print('loading...')

        spectr.awg.ch12.load_waveform('delay',
                                  np.concatenate((delay_w1,delay_w2)),
                                  np.concatenate((delay_markersArray,delay_markersArray)))
        spectr.awg.ch12.load_waveform('In_pos',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((markersArray,markersArray)))
        spectr.awg.ch12.load_waveform('In_noacq',
                                  np.concatenate((I_in_pos,Q_in_pos)),
                                  np.concatenate((noacq_markersArray,noacq_markersArray)))

        print('loaded')

        spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
        spectr.awg.ch12.create_sequence('sequence_bsub',['In_noacq','In_pos'],[200,210])
        start_time = time()

        if bsub ==True:
            nacq = spectr.NumberOfAcquisitions()
            spectr.NumberOfAcquisitions(200)
            spectr.awg.ch12.init_channel('sequence_bsub')
            I_bg, Q_bg, timeI, timeQ = spectr.IQ_data_averaged()
            I_bgs.append(I_bg)
            Q_bgs.append(Q_bg)
            plot_IQ(np.multiply(downsample(timeI,window),1e9),downsample(I_bg,window),downsample(Q_bg,window))
            spectr.NumberOfAcquisitions(nacq)
            #for t_sleep in range(int(python_delay)): sleep(1)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_I_bgs.txt'%(current_field*1e3,sgs.frequency()*1e-6,SRT),I_bgs)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_Q_bgs.txt'%(current_field*1e3,sgs.frequency()*1e-6,SRT),Q_bgs)
        else:
            spectr.awg.ch12.init_channel('sequence_pos')
            #for t_sleep in range(python_delay): sleep(1)
            I_raw, Q_raw, timeI, timeQ = spectr.IQ_data_raw()
            print('Time elapsed (s):',time()-start_time)

            #Demodulate from intermediate carrier frequency
            t_demod = np.add(wfg.time(I_raw[0], spectr.dig.SampleRate()),acq_wait)
            I_demod, Q_demod = wfg.signal_demod(np.mean(I_raw,axis=0)-I_bgs[shot-1], np.mean(Q_raw,axis = 0)-Q_bgs[shot-1],
                                                t_demod, w)

            #Downsample and average
            t = np.multiply(downsample(timeI,window),1e9)
            I = downsample(I_demod,window)
            Q = downsample(Q_demod,window)

            mag = downsample(np.mean(np.sqrt((I_raw-I_bgs[shot-1])**2+(Q_raw-Q_bgs[shot-1])**2),axis = 0),window)

            Is.append(I)
            Qs.append(Q)
            mags.append(mag)
            mag_int.append(np.mean(mag[start:stop]))
            I_int.append(np.mean(I[start:stop]))
            Q_int.append(np.mean(Q[start:stop]))


            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_Is.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT),Is)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_Qs.txt'   %(current_field*1e3,sgs.frequency()*1e-6,SRT),Qs)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_mags.txt' %(current_field*1e3,sgs.frequency()*1e-6,SRT),mags)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_waits.txt'%(current_field*1e3,sgs.frequency()*1e-6,SRT),waits)
            np.savetxt(folder+'\\'+name+'%.3fmT%.3fMHz_%isSRT_T2_t.txt'    %(current_field*1e3,sgs.frequency()*1e-6,SRT),t)

            #Plot
            plot_pickle(waits[0:shot],mag_int,I_int)
            #plot_pickle(waits[0:shot],I_int,I_int)

            plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
        shot+=1

    plt.figure(figsize = (6,5))
    try:
        T2_stretchedfit(mag_int,waits,noise_floor = np.mean(mag[-50:]),title = 'Magnitude, B = %.3fmT, f = %.3fMHz'%(current_field*1e3,sgs.frequency()*1e-6),
               save = False,filename = folder+'\\'+name+' %.3fmT%.3fMHz_T2_mag.pdf' %(current_field*1e3,sgs.frequency()*1e-6),show =False)
        #plt.xlim(0,520)
        #plt.ylim(0,None)
        plt.tight_layout()
        plt.savefig(folder+'\\'+name+' %.3fmT%.3fMHz_mag_T2.pdf'%(current_field*1e3,sgs.frequency()*1e-6))
    except:
        pass
    np.savetxt(folder+'\\'+name+' %.3fmT%.3fMHz_T2_int.txt'%(current_field*1e3,sgs.frequency()*1e-6),np.transpose([waits,I_int,Q_int,mag_int]))
    print("Done!")

def load_fakeecho():

    Pi =10e-6
    SR = 1.28e9/(2**5)
    wfg.sample_rate = SR
    pulse_amplitude = 0.05
    phase =0
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(70e-6,0)
    d2 = wfg.heterodyne_delay(5000e-6,0)
    piebytwo = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase)
    piebytwo_negative = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+180)

    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR + 100e-9
    acq_wait = 100e-9

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,gate_len))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    marker_digitizer = marker_acq

    I_in_pos, Q_in_pos, marker_gate, marker_acq, marker_digitizer = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate, marker_acq, marker_digitizer)
    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    #Echo detection waveform - negative piebytwo

    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_negative,w = w,t0=0)
    I_in_neg, Q_in_neg, marker_gate, marker_acq = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate, marker_acq)
    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
    ###############################################################
    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms
    print('loading...')

    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))

    spectr.awg.ch12.create_sequence('fake_echo_pos',['In_pos'],[1])
    spectr.awg.ch12.create_sequence('fake_echo_neg',['In_neg'],[1])
    print('loaded')

def run_fakeecho_SNR(title = "B = %.3fmT, f = %.3fMHz"%(i3d.field()*1e3,sgs.frequency()*1e-6),save = False,extra_name = "",plot=True, name = None, folder = None):
    SNR_I,SNR_Q,SNR_mag,SNR_I_avg,SNR_Q_avg,SNR_mag_avg = [],[],[],[],[],[]
    acq_wait = 100e-9
    w = np.pi*2*1e6*0

    # Run sequences: positive and negative for phase cycling
    start_time = time()
    spectr.awg.ch12.init_channel('fake_echo_neg')
    n_I, n_Q, timeI, timeQ = spectr.IQ_data_raw()
    #print('Time elapsed (s):',time()-start_time)

    start_time = time()

    spectr.awg.ch12.init_channel('fake_echo_pos')
    p_I, p_Q, timeI, timeQ = spectr.IQ_data_raw()
    #print('Time elapsed (s):',time()-start_time)

    window = 12

    raw_I = np.subtract(p_I, n_I)
    raw_Q = np.subtract(p_Q, n_Q)
    raw_mag = (raw_I**2+raw_Q**2)**0.5

    PhasedI = np.mean(raw_I,axis = 0)
    PhasedQ = np.mean(raw_Q,axis = 0)
    Phasedmag = np.mean(raw_mag,axis = 0)

    # Redefine mag to zero the noise floor
    raw_mag = raw_mag-Phasedmag[10*window:window*250].mean(axis = 0)
    Phasedmag = np.mean(raw_mag,axis = 0)

    for n in range(len(raw_I)-1):
        SNR_I.append((raw_I[n+1][window*310:window*365]**2).mean()/(raw_I[n+1][:window*250]**2).mean())
        SNR_I_avg.append((raw_I[:n+1].mean(axis = 0)[window*310:window*365]**2).mean()/(raw_I[:n+1].mean(axis = 0)[:window*250]**2).mean())

        SNR_Q.append((raw_Q[n+1][window*310:window*365]**2).mean()/(raw_Q[n+1][:window*250]**2).mean())
        SNR_Q_avg.append((raw_Q[:n+1].mean(axis = 0)[window*310:window*365]**2).mean()/(raw_Q[:n+1].mean(axis = 0)[:window*250]**2).mean())

        SNR_mag.append((raw_mag[n+1][window*310:window*365]**2).mean()/(raw_mag[n+1][:window*250]**2).mean())
        SNR_mag_avg.append((raw_mag[:n+1].mean(axis = 0)[window*310:window*365]**2).mean()/(raw_mag[:n+1].mean(axis = 0)[:window*250]**2).mean())

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

    #Downsample and average

    t = np.multiply(downsample(timeI,window),1e9)
    I = np.array(downsample(I_demod,window))
    Q = np.array(downsample(Q_demod,window))
    mag = np.array(downsample(Phasedmag,window))

    if plot ==True:
        frequency = sgs.frequency.get()
        current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5
        plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))

        plt.figure(figsize = (7,5))

        shots = np.linspace(1,len(SNR_mag),len(SNR_mag))

        plt.plot(shots,SNR_mag,"o",label = "shot")
        plt.plot(shots,SNR_mag_avg,"o",label = "cumulative average")

        #fit
        #func = lambda y,x: y[0]*x
        #est,fine,data_fit = fit_function([20],func,shots,SNR_mag_avg)
        #plt.plot(np.linspace(0,len(SNR_mag)+1,len(SNR_mag)+2),func(est,np.linspace(0,len(SNR_mag)+1,len(SNR_mag)+2)),label = "fit")

        plt.plot(np.linspace(0,len(SNR_mag)+1,len(SNR_mag)),np.median(SNR_mag)*np.ones(len(shots)),label = "median SNR = %.1f"%np.median(SNR_mag))

        plt.ylabel("SNR")
        plt.xlabel("Shots")
        #plt.xticks(np.linspace(0,20,11))
        plt.ylim([0,None])
        #plt.ylim([0,10*np.median(SNR_mag)])
        plt.xlim([0,len(SNR_mag)])
        plt.legend()
        plt.title(extra_name+title)
        plt.tight_layout()

        if save ==True:

            filename = folder+"\\"+name+extra_name+"%.2fmT%.3fMHz_fakeecho_SNR"%   (current_field*1e3,sgs.frequency()*1e-6)
            plt.savefig(filename+".pdf")
            np.savetxt(filename+".txt",np.transpose([SNR_I,SNR_Q,SNR_mag,SNR_I_avg,SNR_Q_avg,SNR_mag_avg]))
        plt.show()

    return(SNR_I,SNR_Q,SNR_mag,SNR_I_avg,SNR_Q_avg,SNR_mag_avg)

def get_SNR(title = "B = %.3fmT, f = %.3fMHz"%(i3d.field()*1e3,sgs.frequency()*1e-6),save = False,extra_name = "",plot=True, name = None, folder = None):

    SNR_I,SNR_Q,SNR_mag,SNR_I_avg,SNR_Q_avg,SNR_mag_avg = [],[],[],[],[],[]
    Pi =10e-6
    SR = 1.28e9/(2**5)
    wfg.sample_rate = SR
    pulse_amplitude = 0.05
    phase =0
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(70e-6,0)
    d2 = wfg.heterodyne_delay(5000e-6,0)
    piebytwo = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase)
    piebytwo_negative = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+180)

    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR + 100e-9
    acq_wait = 100e-9

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,gate_len))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    marker_digitizer = marker_acq

    I_in_pos, Q_in_pos, marker_gate, marker_acq, marker_digitizer = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate, marker_acq, marker_digitizer)
    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    #Echo detection waveform - negative piebytwo

    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_negative,w = w,t0=0)
    I_in_neg, Q_in_neg, marker_gate, marker_acq = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate, marker_acq)
    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
    ###############################################################
    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms
    print('loading...')

    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))
    print('loaded')

    spectr.awg.ch12.create_sequence('fake_echo_pos',['In_pos'],[1])
    spectr.awg.ch12.create_sequence('fake_echo_neg',['In_neg'],[1])

    # Run sequences: positive and negative for phase cycling
    start_time = time()
    spectr.awg.ch12.init_channel('fake_echo_neg')
    n_I, n_Q, timeI, timeQ = spectr.IQ_data_raw()
    print('Time elapsed (s):',time()-start_time)

    start_time = time()

    spectr.awg.ch12.init_channel('fake_echo_pos')
    p_I, p_Q, timeI, timeQ = spectr.IQ_data_raw()
    print('Time elapsed (s):',time()-start_time)

    window = 12

    raw_I = np.subtract(p_I, n_I)
    raw_Q = np.subtract(p_Q, n_Q)
    raw_mag = (raw_I**2+raw_Q**2)**0.5

    PhasedI = np.mean(raw_I,axis = 0)
    PhasedQ = np.mean(raw_Q,axis = 0)
    Phasedmag = np.mean(raw_mag,axis = 0)

    # Redefine mag to zero the noise floor
    raw_mag = raw_mag-Phasedmag[10*window:window*250].mean(axis = 0)
    Phasedmag = np.mean(raw_mag,axis = 0)

    for n in range(len(raw_I)-1):
        SNR_I.append((raw_I[n+1][window*310:window*365]**2).mean()/(raw_I[n+1][:window*250]**2).mean())
        SNR_I_avg.append((raw_I[:n+1].mean(axis = 0)[window*310:window*365]**2).mean()/(raw_I[:n+1].mean(axis = 0)[:window*250]**2).mean())

        SNR_Q.append((raw_Q[n+1][window*310:window*365]**2).mean()/(raw_Q[n+1][:window*250]**2).mean())
        SNR_Q_avg.append((raw_Q[:n+1].mean(axis = 0)[window*310:window*365]**2).mean()/(raw_Q[:n+1].mean(axis = 0)[:window*250]**2).mean())

        SNR_mag.append((raw_mag[n+1][window*310:window*365]**2).mean()/(raw_mag[n+1][:window*250]**2).mean())
        SNR_mag_avg.append((raw_mag[:n+1].mean(axis = 0)[window*310:window*365]**2).mean()/(raw_mag[:n+1].mean(axis = 0)[:window*250]**2).mean())

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

    #Downsample and average

    t = np.multiply(downsample(timeI,window),1e9)
    I = np.array(downsample(I_demod,window))
    Q = np.array(downsample(Q_demod,window))
    mag = np.array(downsample(Phasedmag,window))

    if plot ==True:
        plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))

        plt.figure(figsize = (7,5))

        shots = np.linspace(1,len(SNR_mag),len(SNR_mag))

        plt.plot(shots,SNR_mag,"o",label = "shot")
        plt.plot(shots,SNR_mag_avg,"o",label = "cumulative average")

        #fit
        #func = lambda y,x: y[0]*x
        #est,fine,data_fit = fit_function([20],func,shots,SNR_mag_avg)
        #plt.plot(np.linspace(0,len(SNR_mag)+1,len(SNR_mag)+2),func(est,np.linspace(0,len(SNR_mag)+1,len(SNR_mag)+2)),label = "fit")

        plt.plot(np.linspace(0,len(SNR_mag)+1,len(SNR_mag)),np.median(SNR_mag)*np.ones(len(shots)),label = "median SNR = %.1f"%np.median(SNR_mag))

        plt.ylabel("SNR")
        plt.xlabel("Shots")
        #plt.xticks(np.linspace(0,20,11))
        plt.ylim([0,None])
        #plt.ylim([0,10*np.median(SNR_mag)])
        plt.xlim([0,len(SNR_mag)])
        plt.legend()
        plt.title(extra_name+title)
        plt.tight_layout()

        if save ==True:
            current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5
            filename = folder+"\\"+name+extra_name+"%.2fmT%.3fMHz_fakeecho_SNR"%   (current_field*1e3,sgs.frequency()*1e-6)
            plt.savefig(filename+".pdf")
            np.savetxt(filename+".txt",np.transpose([SNR_I,SNR_Q,SNR_mag,SNR_I_avg,SNR_Q_avg,SNR_mag_avg]))
        plt.show()

    return(SNR_I,SNR_Q,SNR_mag,SNR_I_avg,SNR_Q_avg,SNR_mag_avg)

def run_single_powersweep(sweep,step,avg,frequency,measure,bandwidth,filename="vna_sweep",folder = '', save = True,
                          plot = True, printsweep = True):
    npts = int(1+(sweep[1]-sweep[0])/step)
    if npts >maxpoints:
        print('Too many points for single sweep! Use stitched sweeps')
        return(None, None, None)
    if printsweep == True: print('Sweep %.3f to %.3f dBm, frequency %.6fGHz, %i points'%(sweep[0]/1e9,sweep[1]/1e9,frequency*1e9,npts))
    vna.timeout.set(10000);
    vna.set('avg',avg);
    vna.set('CW freq',frequency);
    vna.set('measure',measure);
    vna.set('npts',npts);
    vna.set('bandwidth',bandwidth);
    vna.set('start',sweep[0]);
    vna.set('stop',sweep[1]);
    power = np.linspace(sweep[0],sweep[1],npts);
    (S21,phase) = vna.trace();
    sweepdata = np.transpose([power,S21,phase])
    if plot == True:
        plt.plot(power,S21)
        plt.xlabel('Power (dBm)')
        plt.ylabel('S21 (dB)')
        plt.title(filename)
        plt.grid(True)
        plt.show()
        plt.clf()

    return(power,S21,phase)

def plot_SNR(t,I,Q,mag,echo_start,echo_stop,end,extra_name,extra_title = "",save = False, name = None, folder = None):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plot_IQ(t,I,Q,show = False)
    plot_IQ(t[echo_start:echo_stop],I[echo_start:echo_stop],Q[echo_start:echo_stop],
            title = extra_title+"B = %.3fmT, f = %.3fMHz, SNR = %.1f"
            %(current_field*1e3,sgs.frequency()*1e-6,((I[echo_start:echo_stop]**2).mean()/(I[end:]**2).mean())),show = False)

    plt.subplot(1, 2, 2)
    plot_mag(t,mag,show = False)
    plt.plot(t[echo_start:echo_stop]/1000,mag[echo_start:echo_stop])
    plt.plot(t[end:]/1000,mag[end:])
    print("SNR = %.1f"%((I[echo_start:echo_stop]**2).mean()/(I[end:]**2).mean()))
    print("Mag SNR = %.1f"%((mag[echo_start:echo_stop]**2).mean()/(mag[end:]**2).mean()))

    if save == True:
        plt.savefig(folder+"\\"+name+extra_name+"%iNCPMG%.2fmT%.3fMHz%iustau_echo.pdf"% (N_CPMG,current_field*1e3,sgs.frequency()*1e-6,wait*1e6))
        np.savetxt(folder+"\\"+name+extra_name+"%iNCPMG%.2fmT%.3fMHz%iustau%iavgs_echo_t.txt"%   (N_CPMG,current_field*1e3,sgs.frequency()*1e-6,wait*1e6,avgs),         t)
        np.savetxt(folder+"\\"+name+extra_name+"%iNCPMG%.2fmT%.3fMHz%iustau%iavgs_echo_I.txt"%   (N_CPMG,current_field*1e3,sgs.frequency()*1e-6,wait*1e6,avgs),  I_phased)
        np.savetxt(folder+"\\"+name+extra_name+"%iNCPMG%.2fmT%.3fMHz%iustau%iavgs_echo_Q.txt"%   (N_CPMG,current_field*1e3,sgs.frequency()*1e-6,wait*1e6,avgs),  Q_phased)
        np.savetxt(folder+"\\"+name+extra_name+"%iNCPMG%.2fmT%.3fMHz%iustau%iavgs_echo_mag.txt"% (N_CPMG,current_field*1e3,sgs.frequency()*1e-6,wait*1e6,avgs),mag_phased)

def run_CPMG_positive(shot_rep,N_CPMG,acq_wait,end_delay,w = 0,window = 250,noacq = False,plot=True,raw=False,bsub=False):
    if bsub == True:
        tn,In,Qn,magn = run_CPMG_bsub(.01,1000,acq_wait,.001,w = w,window = 1,noacq = False,plot=False)
    start_time = time()
    if N_CPMG ==0:spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
    else:
        if noacq == True: spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','ref_noacq','delay'],[1,N_CPMG,int(shot_rep/end_delay)])
        else: spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','refocused','delay'],[1,N_CPMG,int(shot_rep/end_delay)])
    # Run positive sequence
    spectr.awg.ch12.init_channel('sequence_pos')
    if raw == False: I, Q, timeI, timeQ = spectr.IQ_data_averaged()
    if raw == True:
        I, Q, timeI, timeQ = spectr.IQ_data_raw()
        if bsub == False:
            I_bckgrd = np.mean(np.mean(I,axis = 0)[-20:])
            Q_bckgrd = np.mean(np.mean(Q,axis = 0)[-20:])
            mag = np.mean(np.sqrt((I - I_bckgrd)**2+(Q - Q_bckgrd)**2),axis = 0)
            I = np.mean(I,axis = 0)
            Q = np.mean(Q,axis = 0)
        if bsub == True:
            I = [np.subtract(i, In) for i in I]
            Q = [np.subtract(q, Qn) for q in Q]
            mag = np.mean(np.sqrt(np.array(I)**2+np.array(Q)**2),axis = 0)
            I = np.mean(I,axis = 0)
            Q = np.mean(Q,axis = 0)
    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(I, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(I, Q,t_demod, w)
    #Downsample and average
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)
    Q = downsample(Q_demod,window)
    if raw == False: mag = np.sqrt(np.array(I)**2+np.array(Q)**2)
    if raw == True: mag = downsample(mag, window)
    if plot==True:
        s = 5
        t1,I1,Q1,mag1 = t[s:],I[s:],Q[s:],mag[s:]
        plot_IQmag(t1,I1,Q1,mag1,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    #Must clear sequences because we want to update the shot rep time
    spectr.awg.ch12.clear_sequence('sequence_pos')
    print('Time elapsed (s):',time()-start_time)
    return(t,I,Q,mag)

def run_CPMG_negative(shot_rep,N_CPMG,acq_wait,end_delay,w = 0,window = 250,noacq = False,plot=True, raw=False, bsub=False):
    start_time = time()
    if N_CPMG ==0: spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','delay'],[1,int(shot_rep/end_delay)])
    else:
        if noacq == True: spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','ref_noacq','delay'],[1,N_CPMG,int(shot_rep/end_delay)])
        else: spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','refocused','delay'],[1,N_CPMG,int(shot_rep/end_delay)])
    # Run negative sequence
    spectr.awg.ch12.init_channel('sequence_neg')
    if raw == False: I, Q, timeI, timeQ = spectr.IQ_data_averaged()
    if raw == True:
        I, Q, timeI, timeQ = spectr.IQ_data_raw()
        if bsub == False:
            I_bckgrd = np.mean(np.mean(I,axis = 0)[-20:])
            Q_bckgrd = np.mean(np.mean(Q,axis = 0)[-20:])
            mag = np.mean(np.sqrt((I - I_bckgrd)**2+(Q - Q_bckgrd)**2),axis = 0)
            I = np.mean(I,axis = 0)
            Q = np.mean(Q,axis = 0)
        if bsub == True:
            I = [np.subtract(i, In) for i in I]
            Q = [np.subtract(q, Qn) for q in Q]
            mag = np.mean(np.sqrt(np.array(I)**2+np.array(Q)**2),axis = 0)
            I = np.mean(I,axis = 0)
            Q = np.mean(Q,axis = 0)
    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(I, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(I, Q,t_demod, w)
    #Downsample and average
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)
    Q = downsample(Q_demod,window)
    if raw == False: mag = np.sqrt(np.array(I)**2+np.array(Q)**2)
    if raw == True: mag = downsample(mag, window)
    if plot==True:
        s = 5
        t1,I1,Q1,mag1 = t[s:],I[s:],Q[s:],mag[s:]
        plot_IQmag(t1,I1,Q1,mag1,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    #Must clear sequences because we want to update the shot rep time
    spectr.awg.ch12.clear_sequence('sequence_neg')
    print('Time elapsed (s):',time()-start_time)
    return(t,I,Q,mag)

def load_CPMG(Pi,pulse_amplitude,wait,wait_CPMG,phase = 0,w = 0,gate_wait = 50e-9,SR = 2e7,end_delay = 1e-3,
              plot_waveforms = False,noacq = False, gaussian=False,pibytwoonly=False,acqpulse = True):
    start_time = time()
    print('loading...')
    spectr.awg.ClearMemory()
    wfg.sample_rate = SR
    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    d1 = wfg.heterodyne_delay(150e-9,0)
    d2 = wfg.heterodyne_delay(wait,0)
    d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
    d4 = wfg.heterodyne_delay(wait_CPMG*0.1,0)
    if acqpulse ==True: acqpulse = wfg.heterodyne_pulse(1,wait_CPMG*1.8,0)
    else: acqpulse = wfg.heterodyne_delay(wait_CPMG*1.8,0)
    if gaussian == False:
        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
        piebytwo_negative = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Pi/2,0)
        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
        if pibytwoonly == True: pie, pie_neg, pie_bsub = piebytwo, piebytwo_negative, piebytwo
    if gaussian == True:
        piebytwo = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi/2,phase)
        piebytwo_negative = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi/2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Pi/2,0)
        d2 = wfg.heterodyne_delay(wait,0)
        d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
        pie = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
        if pibytwoonly == True: pie, pie_neg, pie_bsub = piebytwo, piebytwo_negative, piebytwo
    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d4,acqpulse,d4,w = w,t0=0)

    gate_len = len(I_in_pos)/SR-2*wait_CPMG + 100e-9
    acq_wait = gate_len +wait-20e-6#100E-9#
    if noacq == True: acq_wait = 100e-9

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,gate_len))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,gate_len+2*wait-acq_wait))

    I_in_pos, Q_in_pos, marker_gate, marker_acq = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate, marker_acq)
    marker_digitizer = marker_acq

    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    #Echo detection waveform - negative piebytwo
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_negative,d2,pie_neg,d4,acqpulse,d4,w = w,t0=0)
    I_in_neg, Q_in_neg ,marker_gate, marker_acq = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate, marker_acq)

    if plot_waveforms == True: wfg.plot_waveforms(I_in_pos,Q_in_pos,marker_gate,marker_acq,SR)
    if plot_waveforms == True: wfg.plot_waveforms(I_in_neg,Q_in_neg,marker_gate,marker_acq,SR)

    #Background subtraction waveform
    I_in_bsub,Q_in_bsub = wfg.heterodyne_combine(d1,piebytwo_bsub,d2,pie_bsub,d3,w = w,t0=0)
    I_in_bsub, Q_in_bsub,marker_gate, marker_acq = wfg.length_correction(I_in_bsub, Q_in_bsub,marker_gate, marker_acq)
    ###############################################################
    # delay waveform for reptime setting
    delay_w1 = wfg.delay(end_delay)

    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_marker_gate)
    ################################################################
    #Refocused echo detection waveform

    I_refocused,Q_refocused = wfg.heterodyne_combine(pie,d4,acqpulse,w = w,t0=0)

    gate_len = len(I_refocused)/SR - wait_CPMG*1.9
    acq_wait = gate_len +wait_CPMG-21.5e-6#100E-9#

    refocused_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    refocused_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    if noacq == True: refocused_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    else: refocused_marker_digitizer = refocused_marker_acq
    I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer = wfg.length_correction(I_refocused, Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)
    refocused_markersArray = spectr.awg.generate_MarkersArray(refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)

    if plot_waveforms == True: wfg.plot_waveforms(I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,SR)
    ################################################################
    #refocusing with no acquisition

    I_ref_noacq,Q_ref_noacq = wfg.heterodyne_combine(pie,w = w,t0=0)

    gate_len = len(I_ref_noacq)/SR + 0e-9
    acq_wait = gate_len +wait_CPMG-11.5e-6#100E-9#

    ref_noacq_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    ref_noacq_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    ref_noacq_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    I_ref_noacq,Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer = wfg.length_correction(I_ref_noacq, Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ref_noacq_markersArray = spectr.awg.generate_MarkersArray(ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ################################################################
    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms

    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_bsub',
                              np.concatenate((I_in_bsub,Q_in_bsub)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('refocused',
                              np.concatenate((I_refocused,Q_refocused)),
                              np.concatenate((refocused_markersArray,refocused_markersArray)))
    spectr.awg.ch12.load_waveform('ref_noacq',
                              np.concatenate((I_ref_noacq,Q_ref_noacq)),
                              np.concatenate((ref_noacq_markersArray,ref_noacq_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))

    print('loaded in %.1fs'%(time()-start_time))

    return(acq_wait,end_delay)

def run_single_sweep(sweep,step,avg,power,measure,bandwidth,filename="vna_sweep",folder = '', save = True, plot = True,name = None):
    npts = int(1+(sweep[1]-sweep[0])/step)
    if npts >maxpoints:
        print('Too many points for single sweep! Use stitched sweeps')
        return(None, None, None)
    print('Sweep %.3f to %.3f GHz, power %i dBm, %i points'%(sweep[0]/1e9,sweep[1]/1e9,power,npts))
    vna.timeout.set(10000);
    vna.set('avg',avg);
    vna.set('power',power);
    vna.set('measure',measure);
    vna.set('npts',npts);
    vna.set('bandwidth',bandwidth);
    vna.set('start',sweep[0]);
    vna.set('stop',sweep[1]);
    frequency = np.linspace(sweep[0],sweep[1],npts);
    (S21,phase) = vna.trace();
    sweepdata = np.transpose([frequency,S21,phase])
    if plot == True:
        plt.plot(frequency,S21)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('S21 (dB)')
        plt.title(filename)
        plt.grid(True)
        if save == True:
            filename = filename+'_%.3f-%.3fMHz,%iavgs,%idBm,%s,%ibw'%(sweep[0]/1e6,sweep[1]/1e6,avg,power,measure,bandwidth)
            np.savetxt(folder+"\\"+filename+'.txt',sweepdata);
            plt.tight_layout()
            plt.savefig(folder+"\\"+filename+'.png')
        plt.show()
        plt.clf()
    elif save == True:
        filename = filename+'_%.3f-%.3fMHz,%iavgs,%idBm,%s,%ibw'%(sweep[0]/1e6,sweep[1]/1e6,avg,power,measure,bandwidth)
        np.savetxt(folder+"\\"+filename+'.txt',sweepdata)

    return(frequency,S21,phase)

def fit_FanoResonance(freq,trace,guess_sigma = 0.1,guess_amp = 5e-5,guess_q = 0.5,return_params = False,folder = '',filename = 'Fano_Fit', plot = True,save = True, peak = "max"):
    #print (guess_amp,guess_q,guess_sigma)
    start,stop = None, None #np.argmax(trace)-500,np.argmax(trace)+500# 27900,28200  #Specifies the window within the data to analyse. Set to None,None if you want the whole window
    Lin_mod = LinearModel()                                         #Linear lmfit model for background offset and slope
    BW_mod = BreitWignerModel()                                     #Breit-Wigner-Fano model
    mod = BW_mod+Lin_mod
    x = freq[start:stop]/1E6                                        #Convert frequencies to MHz
    trace = (10**(trace/10))                                        #Convert decibel data to linear
    y = trace[start:stop]

    if peak == "min": guess_centre = np.argmin(y)
    elif peak == "max": guess_centre = np.argmax(y)
    pars = BW_mod.guess(y, x=x)                                     #Initialize fit params
    pars += Lin_mod.guess(y,x=x, slope = 0, vary = False)
    pars['center'].set(value=x[guess_centre], vary=True, expr='')   #Use numpy to find the highest transmission value. Corresponding frequency is used as a guess for the centre frequency
    pars['sigma'].set(value=guess_sigma, vary=True, expr='')                #Linewidth
    pars['q'].set(value=guess_q, vary=True, expr='')                      #Fano factor (asymmetry term). q=0 gives a Lorentzian
    pars['amplitude'].set(value=guess_amp, vary=True, expr='')          #Amplitude

    out  = mod.fit(y,pars,x=x)
    #print (out.fit_report())
    #print (out.params['amplitude'].value,out.params['q'].value,out.params['sigma'].value)
    sigma = out.params['sigma']
    centre = out.params['center']
    q = out.params['q']
    amplitude = out.params['amplitude']
    if plot == True:
        #print(out.params['amplitude'],out.params['q'],out.params['sigma'])
        plt.plot(x,y, color = 'orange', label = 'Data')
        plt.plot(x, out.best_fit, color = 'darkslateblue',label = 'Fano fit:\nC = %.3f MHz\nQ = %i'%(centre.value,centre.value/sigma.value))
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("S21 magnitude (mW)")
        plt.legend()
        if save==True:
            filename = filename+'_%.3f-%.3fMHz,%iavgs,%idBm,%s,%ibw'%(sweep[0]/1e6,sweep[1]/1e6,avg,power,measure,bandwidth)
            plt.tight_layout()
            plt.savefig(folder+"\\"+filename+'.pdf')
    fwhm = out.params['fwhm']
    if return_params == True:
        if np.isnan(sigma.value) ==False: return(sigma.value,centre.value,centre.value/sigma.value,amplitude.value,q.value)
        else: return (1,1,1,1,1)
    else: return(sigma.value,centre.value,centre.value/sigma.value)       #Returns linewidth in GHz, centre in GHz and Q factor

def VNA_mode():
    tenma.set_voltage(12)

def SPEC_mode():
    tenma.set_voltage(0)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def run_avgd_singleshot(Pi,pulse_amplitude,wait,avgs,avgd_shot_rep,phase=0,plot = False,shot_rep = 0.1,N_CPMG = 0,PC = False):
    numrec = spectr.NumRecordsPerAcquisition()
    numacq = spectr.NumberOfAcquisitions()


    acq_wait,end_delay = load_CPMG(Pi,pulse_amplitude,wait,wait,phase,plot_waveforms = False)
    I_avg,Q_avg,mag_avg = [],[],[]
    shot = 1

    if PC==False:
        # IMPORTANT: use following parameters which override the digitizer
        spectr.NumRecordsPerAcquisition(10)      # Specifies the number of records in the acquisition.
        spectr.NumberOfAcquisitions(10)          # Specifies the number of acquistions
        acq_wait,end_delay = load_CPMG(Pi,pulse_amplitude,wait*5,wait*5,phase,plot_waveforms = False)
        tn,In,Qn,magn = run_CPMG_bsub(0.001,N_CPMG,acq_wait,end_delay,noacq = False,window = 250,plot = False)
        acq_wait,end_delay = load_CPMG(Pi,pulse_amplitude,wait,wait,phase,plot_waveforms = False)
        for n in range(int(avgd_shot_rep*100)): sleep(0.01)
        spectr.NumRecordsPerAcquisition(1)      # Specifies the number of records in the acquisition.
        spectr.NumberOfAcquisitions(1)          # Specifies the number of acquistions

    if N_CPMG == 0:
        spectr.NumRecordsPerAcquisition(1)
        spectr.NumberOfAcquisitions(1)
        print('\nTotal number of averages (NumRecordsPerAcquisition x NumberOfAcquisitions): ',spectr.dig.NumberOfAverages())

    for avg in range(avgs):
        #print('\nAvg %i of %i\n'%(shot,avgs))
        if N_CPMG ==0:

            t,I,Q,mag = run_CPMG_positive(avgd_shot_rep,N_CPMG,acq_wait,end_delay,noacq = False,window = 250,plot = False)

            if PC == True:
                tn,In,Qn,magn = run_CPMG_negative(avgd_shot_rep,N_CPMG,acq_wait,end_delay,noacq = False,window = 250,plot = False)

        else:

            t,I,Q,mag = run_CPMG_positive(avgd_shot_rep,N_CPMG,acq_wait,end_delay,noacq = False,window = 250,plot = False)

            if PC == True: tn,In,Qn,magn = run_CPMG_negative(avgd_shot_rep,N_CPMG,acq_wait,end_delay,noacq = False,window = 250,plot = False)

        I_phased,Q_phased = 0.5*np.subtract(I,In),0.5*np.subtract(Q,Qn)
        mag_phased = (I_phased**2+Q_phased**2)**0.5
        I_avg.append(I_phased)
        Q_avg.append(Q_phased)
        mag_avg.append(mag_phased)

        if plot == True:
            plot_pickle(t,np.mean(I_avg,axis=0),np.mean(Q_avg,axis=0))
            plot_IQmag(t,np.mean(I_avg,axis=0),np.mean(Q_avg,axis=0),np.mean(mag_avg,axis=0))
            plt.show()
        shot+=1
        plt.clf()
    spectr.NumRecordsPerAcquisition(numrec)
    spectr.NumberOfAcquisitions(numacq)
    print('\nTotal number of averages (NumRecordsPerAcquisition x NumberOfAcquisitions): ',spectr.dig.NumberOfAverages())
    return(t,I_avg,Q_avg,mag_avg)

def run_CPMG_bsub(shot_rep,N_CPMG,acq_wait,end_delay,w = 0,window = 250,noacq = False,plot=True):
    start_time = time()

    if N_CPMG ==0:
        spectr.awg.ch12.create_sequence('sequence_bsub',['In_bsub','delay'],[1,int(shot_rep/end_delay)])
    else:
        if noacq == True: spectr.awg.ch12.create_sequence('sequence_bsub',['In_bsub','ref_noacq','delay'],[1,N_CPMG,int(shot_rep/end_delay)])
        else: spectr.awg.ch12.create_sequence('sequence_bsub',['In_bsub','refocused','delay'],[1,N_CPMG,int(shot_rep/end_delay)])
    # Run positive sequence
    spectr.awg.ch12.init_channel('sequence_bsub')
    I, Q, timeI, timeQ = spectr.IQ_data_averaged()

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(I, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(I, Q,t_demod, w)

    #Downsample and average
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)
    Q = downsample(Q_demod,window)
    mag = np.sqrt(np.array(I)**2+np.array(Q)**2)


    if plot==True:
        #Plot
        s = 5
        t1,I1,Q1,mag1 = t[s:],I[s:],Q[s:],mag[s:]
        plot_IQmag(t1,I1,Q1,mag1,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))

    #Must clear sequences because we want to update the shot rep time
    spectr.awg.ch12.clear_sequence('sequence_bsub')

    print('Time elapsed (s):',time()-start_time)

    return(t,I,Q,mag)

def load_CPMG_weak_excitation(Piby2, Piby2Amp, Pi,pulse_amplitude,wait,wait_CPMG,phase = 0,w = 0,gate_wait = 50e-9,SR = 2e7,
                              end_delay = 1e-3, plot_waveforms = False,noacq = False, gaussian=False, acqpulse=False):
    start_time = time()
    print('loading...')
    spectr.awg.ClearMemory()
    wfg.sample_rate = SR
    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    d1 = wfg.heterodyne_delay(150e-9,0)
    if gaussian == False:
        piebytwo = wfg.heterodyne_pulse(Piby2Amp,Piby2,phase)
        piebytwo_negative = wfg.heterodyne_pulse(Piby2Amp,Piby2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Piby2,0)
        d2 = wfg.heterodyne_delay(wait,0)
        d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
    if gaussian == True:
        piebytwo = wfg.heterodyne_gaussian_pulse(Piby2Amp,Piby2,phase)
        piebytwo_negative = wfg.heterodyne_gaussian_pulse(Piby2Amp,Piby2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Piby2,0)
        d2 = wfg.heterodyne_delay(wait,0)
        d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
        pie = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
    ###############################################################
    #Echo detection waveform - positive piebytwo
    d4 = wfg.heterodyne_delay(wait_CPMG*0.1,0)
    if acqpulse ==True: acqpulse = wfg.heterodyne_pulse(1,wait_CPMG*1.8,0)
    else: acqpulse = wfg.heterodyne_delay(wait_CPMG*1.8,0)

    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d4,acqpulse,d4,w = w,t0=0)

    gate_len = len(I_in_pos)/SR-2*wait_CPMG + 100e-9
    acq_wait = gate_len +wait-20e-6#100E-9#

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,gate_len))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,gate_len+2*wait-acq_wait))

    I_in_pos, Q_in_pos, marker_gate, marker_acq = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate, marker_acq)
    marker_digitizer = marker_acq

    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    #Echo detection waveform - negative piebytwo
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_negative,d2,pie_neg,d3,w = w,t0=0)
    I_in_neg, Q_in_neg ,marker_gate, marker_acq = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate, marker_acq)

    if plot_waveforms == True: wfg.plot_waveforms(I_in_pos,Q_in_pos,marker_gate,marker_acq,SR)
    if plot_waveforms == True: wfg.plot_waveforms(I_in_neg,Q_in_neg,marker_gate,marker_acq,SR)

    #Background subtraction waveform
    I_in_bsub,Q_in_bsub = wfg.heterodyne_combine(d1,piebytwo_bsub,d2,pie_bsub,d3,w = w,t0=0)
    I_in_bsub, Q_in_bsub,marker_gate, marker_acq = wfg.length_correction(I_in_bsub, Q_in_bsub,marker_gate, marker_acq)
    ###############################################################
    # delay waveform for reptime setting
    delay_w1 = wfg.delay(end_delay)

    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_marker_gate)
    ################################################################
    #Refocused echo detection waveform

    I_refocused,Q_refocused = wfg.heterodyne_combine(pie,w = w,t0=0)

    gate_len = len(I_refocused)/SR + 0e-9
    acq_wait = gate_len +wait_CPMG-21.5e-6#100E-9#

    refocused_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    refocused_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    if noacq == True: refocused_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    else: refocused_marker_digitizer = refocused_marker_acq
    I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer = wfg.length_correction(I_refocused, Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)
    refocused_markersArray = spectr.awg.generate_MarkersArray(refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)

    if plot_waveforms == True: wfg.plot_waveforms(I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,SR)
    ################################################################
    #refocusing with no acquisition

    I_ref_noacq,Q_ref_noacq = wfg.heterodyne_combine(pie,w = w,t0=0)

    gate_len = len(I_ref_noacq)/SR + 0e-9
    acq_wait = gate_len +wait_CPMG-11.5e-6#100E-9#

    ref_noacq_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    ref_noacq_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    ref_noacq_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    I_ref_noacq,Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer = wfg.length_correction(I_ref_noacq, Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ref_noacq_markersArray = spectr.awg.generate_MarkersArray(ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ################################################################
    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms

    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_bsub',
                              np.concatenate((I_in_bsub,Q_in_bsub)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('refocused',
                              np.concatenate((I_refocused,Q_refocused)),
                              np.concatenate((refocused_markersArray,refocused_markersArray)))
    spectr.awg.ch12.load_waveform('ref_noacq',
                              np.concatenate((I_ref_noacq,Q_ref_noacq)),
                              np.concatenate((ref_noacq_markersArray,ref_noacq_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))

    print('loaded in %.1fs'%(time()-start_time))

    return(acq_wait,end_delay)

def load_CPMG_Rabi(amp0,Pi,pulse_amplitude,wait,wait_CPMG,phase = 0,w = 0,gate_wait = 50e-9,SR = 2e7,end_delay = 1e-3,
              plot_waveforms = False,noacq = False, gaussian=False,pibytwoonly=False,acqpulse = True):
    start_time = time()
    print('loading...')
    spectr.awg.ClearMemory()
    wfg.sample_rate = SR
    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    d1 = wfg.heterodyne_delay(150e-9,0)
    d2 = wfg.heterodyne_delay(wait,0)
    d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
    d4 = wfg.heterodyne_delay(wait_CPMG*0.1,0)
    if acqpulse ==True: acqpulse = wfg.heterodyne_pulse(1,wait_CPMG*1.8,0)
    else: acqpulse = wfg.heterodyne_delay(wait_CPMG*1.8,0)
    if gaussian == False:
        piebytwo = wfg.heterodyne_pulse(amp0,Pi/2,phase)
        piebytwo_negative = wfg.heterodyne_pulse(amp0,Pi/2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Pi/2,0)
        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
        if pibytwoonly == True: pie, pie_neg, pie_bsub = piebytwo, piebytwo_negative, piebytwo
    if gaussian == True:
        piebytwo = wfg.heterodyne_gaussian_pulse(amp0,Pi/2,phase)
        piebytwo_negative = wfg.heterodyne_gaussian_pulse(amp0,Pi/2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Pi/2,0)
        d2 = wfg.heterodyne_delay(wait,0)
        d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
        pie = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
        if pibytwoonly == True: pie, pie_neg, pie_bsub = piebytwo, piebytwo_negative, piebytwo
    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d4,acqpulse,d4,w = w,t0=0)

    gate_len = len(I_in_pos)/SR-2*wait_CPMG + 100e-9
    acq_wait = gate_len +wait-20e-6#100E-9#
    if noacq == True: acq_wait = 100e-9

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,gate_len))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,gate_len+2*wait-acq_wait))

    I_in_pos, Q_in_pos, marker_gate, marker_acq = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate, marker_acq)
    marker_digitizer = marker_acq

    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    #Echo detection waveform - negative piebytwo
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_negative,d2,pie_neg,d4,acqpulse,d4,w = w,t0=0)
    I_in_neg, Q_in_neg ,marker_gate, marker_acq = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate, marker_acq)

    if plot_waveforms == True: wfg.plot_waveforms(I_in_pos,Q_in_pos,marker_gate,marker_acq,SR)
    if plot_waveforms == True: wfg.plot_waveforms(I_in_neg,Q_in_neg,marker_gate,marker_acq,SR)

    #Background subtraction waveform
    I_in_bsub,Q_in_bsub = wfg.heterodyne_combine(d1,piebytwo_bsub,d2,pie_bsub,d3,w = w,t0=0)
    I_in_bsub, Q_in_bsub,marker_gate, marker_acq = wfg.length_correction(I_in_bsub, Q_in_bsub,marker_gate, marker_acq)
    ###############################################################
    # delay waveform for reptime setting
    delay_w1 = wfg.delay(end_delay)

    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_marker_gate)
    ################################################################
    #Refocused echo detection waveform

    I_refocused,Q_refocused = wfg.heterodyne_combine(pie,d4,acqpulse,w = w,t0=0)

    gate_len = len(I_refocused)/SR - wait_CPMG*1.9
    acq_wait = gate_len +wait_CPMG-21.5e-6#100E-9#

    refocused_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    refocused_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    if noacq == True: refocused_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    else: refocused_marker_digitizer = refocused_marker_acq
    I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer = wfg.length_correction(I_refocused, Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)
    refocused_markersArray = spectr.awg.generate_MarkersArray(refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)

    if plot_waveforms == True: wfg.plot_waveforms(I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,SR)
    ################################################################
    #refocusing with no acquisition

    I_ref_noacq,Q_ref_noacq = wfg.heterodyne_combine(pie,w = w,t0=0)

    gate_len = len(I_ref_noacq)/SR + 0e-9
    acq_wait = gate_len +wait_CPMG-11.5e-6#100E-9#

    ref_noacq_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    ref_noacq_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    ref_noacq_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    I_ref_noacq,Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer = wfg.length_correction(I_ref_noacq, Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ref_noacq_markersArray = spectr.awg.generate_MarkersArray(ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ################################################################
    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms

    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_bsub',
                              np.concatenate((I_in_bsub,Q_in_bsub)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('refocused',
                              np.concatenate((I_refocused,Q_refocused)),
                              np.concatenate((refocused_markersArray,refocused_markersArray)))
    spectr.awg.ch12.load_waveform('ref_noacq',
                              np.concatenate((I_ref_noacq,Q_ref_noacq)),
                              np.concatenate((ref_noacq_markersArray,ref_noacq_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))

    print('loaded in %.1fs'%(time()-start_time))

    return(acq_wait,end_delay)

def run_CPMG_positive_LED(shot_rep,N_CPMG,acq_wait,end_delay,NLED = 0,w = 0,window = 250,noacq = False,plot=True):
    start_time = time()

    if N_CPMG ==0:
        #print('N_CPMG = 0, running no refocusing pulses')
        if NLED == 0 : spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','delay'],[1,int(shot_rep/end_delay)])
        else: spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','LED','delay'],[1,NLED,int(shot_rep/end_delay)])

    else:
        if NLED == 0 : spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','refocused','delay'],[1,N_CPMG,int(shot_rep/end_delay)])
        else: spectr.awg.ch12.create_sequence('sequence_pos',['In_pos','refocused','LED','delay'],[1,N_CPMG,NLED,int(shot_rep/end_delay)])

    # Run negative sequence
    spectr.awg.ch12.init_channel('sequence_pos')
    I, Q, timeI, timeQ = spectr.IQ_data_averaged()

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(I, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(I, Q,t_demod, w)

    #Downsample and average
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)
    Q = downsample(Q_demod,window)
    mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

    if plot==True:
        #Plot
        s = 5
        t1,I1,Q1,mag1 = t[s:],I[s:],Q[s:],mag[s:]
        plot_IQmag(t1,I1,Q1,mag1,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))

    #Must clear sequences because we want to update the shot rep time
    spectr.awg.ch12.clear_sequence('sequence_pos')

    print('Time elapsed (s):',time()-start_time)

    return(t,I,Q,mag)

def run_CPMG_negative_LED(shot_rep,N_CPMG,acq_wait,end_delay,NLED = 0,w = 0,window = 250,noacq = False,plot=True):
    start_time = time()

    if N_CPMG ==0:
        #print('N_CPMG = 0, running no refocusing pulses')
        if NLED == 0 : spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','delay'],[1,int(shot_rep/end_delay)])
        else: spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','LED','delay'],[1,NLED,int(shot_rep/end_delay)])

    else:
        if NLED == 0 : spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','refocused','delay'],[1,N_CPMG,int(shot_rep/end_delay)])
        else: spectr.awg.ch12.create_sequence('sequence_neg',['In_neg','refocused','LED','delay'],[1,N_CPMG,NLED,int(shot_rep/end_delay)])

    # Run negative sequence
    spectr.awg.ch12.init_channel('sequence_neg')
    I, Q, timeI, timeQ = spectr.IQ_data_averaged()

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(I, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(I, Q,t_demod, w)

    #Downsample and average
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)
    Q = downsample(Q_demod,window)
    mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

    if plot==True:
        #Plot
        s = 5
        t1,I1,Q1,mag1 = t[s:],I[s:],Q[s:],mag[s:]
        plot_IQmag(t1,I1,Q1,mag1,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))

    #Must clear sequences because we want to update the shot rep time
    spectr.awg.ch12.clear_sequence('sequence_neg')

    print('Time elapsed (s):',time()-start_time)

    return(t,I,Q,mag)

def load_CPMG_LED(Pi,pulse_amplitude,wait,wait_CPMG,phase = 0,w = 0,gate_wait = 50e-9,SR = 2e7,end_delay = 1e-3,
                  LED_shot_rep = 100e-6,LED_marker_duration = 1e-6, plot_waveforms = False,noacq = False):

    start_time = time()
    print('loading...')
    spectr.awg.ClearMemory()
    wfg.sample_rate = SR
    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    d1 = wfg.heterodyne_delay(150e-9,0)
    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_negative = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)
    piebytwo_bsub = wfg.heterodyne_delay(Pi/2,0)
    d2 = wfg.heterodyne_delay(wait,0)
    d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
    pie_bsub = pie
    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,w = w,t0=0)

    gate_len = len(I_in_pos)/SR-2*wait_CPMG + 100e-9
    acq_wait = gate_len +wait-20e-6#100E-9#

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,gate_len))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,gate_len+2*wait-acq_wait))

    I_in_pos, Q_in_pos, marker_gate, marker_acq = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate, marker_acq)
    marker_digitizer = marker_acq

    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    #Echo detection waveform - negative piebytwo
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_negative,d2,pie_neg,d3,w = w,t0=0)
    I_in_neg, Q_in_neg ,marker_gate, marker_acq = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate, marker_acq)

    if plot_waveforms == True: wfg.plot_waveforms(I_in_pos,Q_in_pos,marker_gate,marker_acq,SR)

    #Background subtraction waveform
    I_in_bsub,Q_in_bsub = wfg.heterodyne_combine(d1,piebytwo_bsub,d2,pie_bsub,d3,w = w,t0=0)
    I_in_bsub, Q_in_bsub,marker_gate, marker_acq = wfg.length_correction(I_in_bsub, Q_in_bsub,marker_gate, marker_acq)
    ###############################################################
    # delay waveform for reptime setting
    delay_w1 = wfg.delay(end_delay)

    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_marker_gate)
    ################################################################
    #Refocused echo detection waveform

    I_refocused,Q_refocused = wfg.heterodyne_combine(pie,w = w,t0=0)

    gate_len = len(I_refocused)/SR + 0e-9
    acq_wait = gate_len +wait_CPMG-21.5e-6#100E-9#

    refocused_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    refocused_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    if noacq == True: refocused_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    else: refocused_marker_digitizer = refocused_marker_acq
    I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer = wfg.length_correction(I_refocused, Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)
    refocused_markersArray = spectr.awg.generate_MarkersArray(refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)

    if plot_waveforms == True: wfg.plot_waveforms(I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,SR)
    ################################################################
    #refocusing with no acquisition

    I_ref_noacq,Q_ref_noacq = wfg.heterodyne_combine(pie,w = w,t0=0)

    gate_len = len(I_ref_noacq)/SR + 0e-9
    acq_wait = gate_len +wait_CPMG-11.5e-6#100E-9#

    ref_noacq_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    ref_noacq_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    ref_noacq_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    I_ref_noacq,Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer = wfg.length_correction(I_ref_noacq, Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ref_noacq_markersArray = spectr.awg.generate_MarkersArray(ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ################################################################
    # LED waveform
    LED_delay = wfg.pulse(0, LED_shot_rep)
    LED_trig = wfg.combine(wfg.pulse(1, LED_marker_duration),wfg.pulse(0, LED_shot_rep-LED_marker_duration))
    marker_LED = spectr.awg.generate_MarkersArray(LED_delay,LED_delay,LED_delay,LED_trig)
    ################################################################

    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms

    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_bsub',
                              np.concatenate((I_in_bsub,Q_in_bsub)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('refocused',
                              np.concatenate((I_refocused,Q_refocused)),
                              np.concatenate((refocused_markersArray,refocused_markersArray)))
    spectr.awg.ch12.load_waveform('ref_noacq',
                              np.concatenate((I_ref_noacq,Q_ref_noacq)),
                              np.concatenate((ref_noacq_markersArray,ref_noacq_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))
    spectr.awg.ch12.load_waveform('LED',
                              np.concatenate((LED_delay,LED_delay)),
                              np.concatenate((marker_LED,marker_LED)))

    print('loaded in %.1fs'%(time()-start_time))

    return(acq_wait,end_delay)

def load_SSADD(Pi,pulse_amplitude,wait,long_wait,phase = 0,w = 0,gate_wait = 50e-9,SR = 2e7,end_delay = 1e-3,
              plot_waveforms = False,noacq = False, gaussian=False,pibytwoonly=False):
    start_time = time()
    print('loading...')
    spectr.awg.ClearMemory()
    wfg.sample_rate = SR
    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    d1 = wfg.heterodyne_delay(150e-9,0)
    if gaussian == False:
        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
        piebytwo_negative = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Pi/2,0)
        d2 = wfg.heterodyne_delay(wait,0)
        d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
        if pibytwoonly == True: pie, pie_neg, pie_bsub = piebytwo, piebytwo_negative, piebytwo
    if gaussian == True:
        piebytwo = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi/2,phase)
        piebytwo_negative = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi/2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Pi/2,0)
        d2 = wfg.heterodyne_delay(wait,0)
        d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
        pie = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
        if pibytwoonly == True: pie, pie_neg, pie_bsub = piebytwo, piebytwo_negative, piebytwo
    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d3,w = w,t0=0)

    gate_len = len(I_in_pos)/SR-2*wait_CPMG + 100e-9
    acq_wait = gate_len +wait-20e-6#100E-9#
    if noacq == True: acq_wait = 100e-9

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,gate_len))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,gate_len+2*wait-acq_wait))

    I_in_pos, Q_in_pos, marker_gate, marker_acq = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate, marker_acq)
    marker_digitizer = marker_acq

    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    #Echo detection waveform - negative piebytwo
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_negative,d2,pie_neg,d3,w = w,t0=0)
    I_in_neg, Q_in_neg ,marker_gate, marker_acq = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate, marker_acq)

    if plot_waveforms == True: wfg.plot_waveforms(I_in_pos,Q_in_pos,marker_gate,marker_acq,SR)
    if plot_waveforms == True: wfg.plot_waveforms(I_in_neg,Q_in_neg,marker_gate,marker_acq,SR)

    #Background subtraction waveform
    I_in_bsub,Q_in_bsub = wfg.heterodyne_combine(d1,piebytwo_bsub,d2,pie_bsub,d3,w = w,t0=0)
    I_in_bsub, Q_in_bsub,marker_gate, marker_acq = wfg.length_correction(I_in_bsub, Q_in_bsub,marker_gate, marker_acq)
    ###############################################################
    # delay waveform for reptime setting
    delay_w1 = wfg.delay(end_delay)

    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_marker_gate)
    ################################################################
    #Refocused echo detection waveform

    I_refocused,Q_refocused = wfg.heterodyne_combine(pie,w = w,t0=0)

    gate_len = len(I_refocused)/SR + 0e-9
    acq_wait = gate_len +wait_CPMG-21.5e-6#100E-9#

    refocused_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    refocused_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    if noacq == True: refocused_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    else: refocused_marker_digitizer = refocused_marker_acq
    I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer = wfg.length_correction(I_refocused, Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)
    refocused_markersArray = spectr.awg.generate_MarkersArray(refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)

    if plot_waveforms == True: wfg.plot_waveforms(I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,SR)
    ################################################################
    #refocusing with no acquisition

    I_ref_noacq,Q_ref_noacq = wfg.heterodyne_combine(pie,w = w,t0=0)

    gate_len = len(I_ref_noacq)/SR + 0e-9
    acq_wait = gate_len +wait_CPMG-11.5e-6#100E-9#

    ref_noacq_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    ref_noacq_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    ref_noacq_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    I_ref_noacq,Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer = wfg.length_correction(I_ref_noacq, Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ref_noacq_markersArray = spectr.awg.generate_MarkersArray(ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ################################################################
    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms

    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_bsub',
                              np.concatenate((I_in_bsub,Q_in_bsub)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('refocused',
                              np.concatenate((I_refocused,Q_refocused)),
                              np.concatenate((refocused_markersArray,refocused_markersArray)))
    spectr.awg.ch12.load_waveform('ref_noacq',
                              np.concatenate((I_ref_noacq,Q_ref_noacq)),
                              np.concatenate((ref_noacq_markersArray,ref_noacq_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))

    print('loaded in %.1fs'%(time()-start_time))

    return(acq_wait,end_delay)

def run_CPMG_positive_sat(shot_rep,N_CPMG,acq_wait,sat_wait,end_delay,w = 0,window = 250,noacq = False,plot=True,raw=False,bsub=False):
    if bsub == True:
        tn,In,Qn,magn = run_CPMG_bsub(.01,1000,acq_wait,.001,w = w,window = 1,noacq = False,plot=False)
    start_time = time()
    if N_CPMG ==0:spectr.awg.ch12.create_sequence('sequence_pos',['Sat_pulse','delay','In_pos','dummy'],[1,int(sat_wait/end_delay),1,1])
    else:
        if noacq == True: spectr.awg.ch12.create_sequence('sequence_pos',['Sat_pulse','delay','In_pos','ref_noacq'],[1,int(sat_wait/end_delay),1,N_CPMG])
        else: spectr.awg.ch12.create_sequence('sequence_pos',['Sat_pulse','delay','In_pos','refocused'],[1,int(sat_wait/end_delay),1,N_CPMG])
    # Run positive sequence
    spectr.awg.ch12.init_channel('sequence_pos')
    if raw == False: I, Q, timeI, timeQ = spectr.IQ_data_averaged()
    if raw == True:
        I, Q, timeI, timeQ = spectr.IQ_data_raw()
        if bsub == False:
            I_bckgrd = np.mean(np.mean(I,axis = 0)[-20:])
            Q_bckgrd = np.mean(np.mean(Q,axis = 0)[-20:])
            mag = np.mean(np.sqrt((I - I_bckgrd)**2+(Q - Q_bckgrd)**2),axis = 0)
            I = np.mean(I,axis = 0)
            Q = np.mean(Q,axis = 0)
        if bsub == True:
            I = [np.subtract(i, In) for i in I]
            Q = [np.subtract(q, Qn) for q in Q]
            mag = np.mean(np.sqrt(np.array(I)**2+np.array(Q)**2),axis = 0)
            I = np.mean(I,axis = 0)
            Q = np.mean(Q,axis = 0)
    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(I, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(I, Q,t_demod, w)
    #Downsample and average
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)
    Q = downsample(Q_demod,window)
    if raw == False: mag = np.sqrt(np.array(I)**2+np.array(Q)**2)
    if raw == True: mag = downsample(mag, window)
    if plot==True:
        s = 5
        t1,I1,Q1,mag1 = t[s:],I[s:],Q[s:],mag[s:]
        plot_IQmag(t1,I1,Q1,mag1,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    #Must clear sequences because we want to update the shot rep time
    spectr.awg.ch12.clear_sequence('sequence_pos')
    print('Time elapsed (s):',time()-start_time)
    return(t,I,Q,mag)

def run_CPMG_negative_sat(shot_rep,N_CPMG,acq_wait,sat_wait,end_delay,w = 0,window = 250,noacq = False,plot=True, raw=False, bsub=False):
    start_time = time()
    if N_CPMG ==0: spectr.awg.ch12.create_sequence('sequence_neg',['Sat_pulse','delay','In_neg','dummy'],[1,int(sat_wait/end_delay),1,1])
    else:
        if noacq == True: spectr.awg.ch12.create_sequence('sequence_neg',['Sat_pulse','delay','In_neg','ref_noacq'],[1,int(sat_wait/end_delay),1,N_CPMG])
        else: spectr.awg.ch12.create_sequence('sequence_neg',['Sat_pulse','delay','In_neg','refocused'],[1,int(sat_wait/end_delay),1,N_CPMG])
    # Run negative sequence
    spectr.awg.ch12.init_channel('sequence_neg')
    if raw == False: I, Q, timeI, timeQ = spectr.IQ_data_averaged()
    if raw == True:
        I, Q, timeI, timeQ = spectr.IQ_data_raw()
        if bsub == False:
            I_bckgrd = np.mean(np.mean(I,axis = 0)[-20:])
            Q_bckgrd = np.mean(np.mean(Q,axis = 0)[-20:])
            mag = np.mean(np.sqrt((I - I_bckgrd)**2+(Q - Q_bckgrd)**2),axis = 0)
            I = np.mean(I,axis = 0)
            Q = np.mean(Q,axis = 0)
        if bsub == True:
            I = [np.subtract(i, In) for i in I]
            Q = [np.subtract(q, Qn) for q in Q]
            mag = np.mean(np.sqrt(np.array(I)**2+np.array(Q)**2),axis = 0)
            I = np.mean(I,axis = 0)
            Q = np.mean(Q,axis = 0)
    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(I, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(I, Q,t_demod, w)
    #Downsample and average
    t = np.multiply(downsample(timeI,window),1e9)
    I = downsample(I_demod,window)
    Q = downsample(Q_demod,window)
    if raw == False: mag = np.sqrt(np.array(I)**2+np.array(Q)**2)
    if raw == True: mag = downsample(mag, window)
    if plot==True:
        s = 5
        t1,I1,Q1,mag1 = t[s:],I[s:],Q[s:],mag[s:]
        plot_IQmag(t1,I1,Q1,mag1,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))
    #Must clear sequences because we want to update the shot rep time
    spectr.awg.ch12.clear_sequence('sequence_neg')
    print('Time elapsed (s):',time()-start_time)
    return(t,I,Q,mag)

def load_CPMG_sat(Pi,pulse_amplitude,wait,wait_CPMG,t_sat,wait_sat,phase = 0,w = 0,gate_wait = 50e-9,SR = 2e7,
                  end_delay = 1e-3,plot_waveforms = False,noacq = False, gaussian=False,pibytwoonly=False,acqpulse = False):
    start_time = time()
    print('loading...')
    spectr.awg.ClearMemory()
    wfg.sample_rate = SR
    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    d1 = wfg.heterodyne_delay(150e-9,0)
    d2 = wfg.heterodyne_delay(wait,0)
    d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
    d4 = wfg.heterodyne_delay(wait_CPMG*0.1,0)
    if acqpulse ==True: acqpulse = wfg.heterodyne_pulse(1,wait_CPMG*1.8,0)
    else: acqpulse = wfg.heterodyne_delay(wait_CPMG*1.8,0)
    if gaussian == False:
        piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
        piebytwo_negative = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Pi/2,0)
        pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
        if pibytwoonly == True: pie, pie_neg, pie_bsub = piebytwo, piebytwo_negative, piebytwo
    if gaussian == True:
        piebytwo = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi/2,phase)
        piebytwo_negative = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi/2,phase+180)
        piebytwo_bsub = wfg.heterodyne_delay(Pi/2,0)
        d2 = wfg.heterodyne_delay(wait,0)
        d3 = wfg.heterodyne_delay(wait_CPMG*2,0)
        pie = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90)
        pie_neg = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+90) # leave identical to pie unless you want to phase cycle the pi pulse!
        pie_bsub = pie
        if pibytwoonly == True: pie, pie_neg, pie_bsub = piebytwo, piebytwo_negative, piebytwo
    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,pie,d4,acqpulse,d4,w = w,t0=0)

    gate_len = len(I_in_pos)/SR-2*wait_CPMG + 100e-9
    acq_wait = gate_len +wait-20e-6#100E-9#
    if noacq == True: acq_wait = 100e-9

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,gate_len))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,gate_len+2*wait-acq_wait))

    I_in_pos, Q_in_pos, marker_gate, marker_acq = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate, marker_acq)
    marker_digitizer = marker_acq

    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    #Echo detection waveform - negative piebytwo
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_negative,d2,pie_neg,d4,acqpulse,d4,w = w,t0=0)
    I_in_neg, Q_in_neg ,marker_gate, marker_acq = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate, marker_acq)

    if plot_waveforms == True: wfg.plot_waveforms(I_in_pos,Q_in_pos,marker_gate,marker_acq,SR)
    if plot_waveforms == True: wfg.plot_waveforms(I_in_neg,Q_in_neg,marker_gate,marker_acq,SR)

    #Background subtraction waveform
    I_in_bsub,Q_in_bsub = wfg.heterodyne_combine(d1,piebytwo_bsub,d2,pie_bsub,d3,w = w,t0=0)
    I_in_bsub, Q_in_bsub,marker_gate, marker_acq = wfg.length_correction(I_in_bsub, Q_in_bsub,marker_gate, marker_acq)
    ###############################################################
    # delay waveform for reptime setting
    delay_w1 = wfg.delay(end_delay)

    delay_w2 = delay_w1
    delay_marker_gate = delay_w1
    delay_w1, delay_w2, delay_marker_gate = wfg.length_correction(delay_w1,delay_w2,delay_marker_gate)
    delay_markersArray = spectr.awg.generate_MarkersArray(delay_marker_gate)
    ################################################################
    #Refocused echo detection waveform

    I_refocused,Q_refocused = wfg.heterodyne_combine(pie,d4,acqpulse,w = w,t0=0)

    gate_len = len(I_refocused)/SR - wait_CPMG*1.9
    acq_wait = gate_len +wait_CPMG-21.5e-6#100E-9#

    refocused_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    refocused_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    if noacq == True: refocused_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    else: refocused_marker_digitizer = refocused_marker_acq
    I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer = wfg.length_correction(I_refocused, Q_refocused,refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)
    refocused_markersArray = spectr.awg.generate_MarkersArray(refocused_marker_gate,refocused_marker_acq,refocused_marker_digitizer)


    sat_pulse = wfg.pulse(1,t_sat)
    I_sat, Q_sat, sat_marker_gate = wfg.length_correction(sat_pulse, sat_pulse, sat_pulse)
    sat_markersArray = spectr.awg.generate_MarkersArray(sat_marker_gate)

    if plot_waveforms == True: wfg.plot_waveforms(I_refocused,Q_refocused,refocused_marker_gate,refocused_marker_acq,SR)
    ################################################################
    #dummy trigger to stop digitiser during single shot mode
    dummy_w1 = wfg.delay(end_delay)

    dummy_w2 = dummy_w1
    dummy_marker_gate = dummy_w1

    dummy_marker_digitiser = wfg.combine(wfg.delay(end_delay/2),wfg.pulse(1,100e-9))

    dummy_w1, dummy_w2, dummy_marker_gate,dummy_marker_acq,dummy_marker_digitiser = wfg.length_correction(dummy_w1,dummy_w2,dummy_marker_gate,dummy_marker_gate,dummy_marker_digitiser)
    dummy_markersArray = spectr.awg.generate_MarkersArray(dummy_marker_gate,dummy_marker_acq,dummy_marker_digitiser)
    ################################################################
    #refocusing with no acquisition

    I_ref_noacq,Q_ref_noacq = wfg.heterodyne_combine(pie,w = w,t0=0)

    gate_len = len(I_ref_noacq)/SR + 0e-9
    acq_wait = gate_len +wait_CPMG-11.5e-6#100E-9#

    ref_noacq_marker_gate = wfg.combine(wfg.pulse(1,gate_len))
    ref_noacq_marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,Pi+2*wait_CPMG-acq_wait))
    ref_noacq_marker_digitizer = wfg.delay(Pi+2*wait_CPMG)
    I_ref_noacq,Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer = wfg.length_correction(I_ref_noacq, Q_ref_noacq,ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ref_noacq_markersArray = spectr.awg.generate_MarkersArray(ref_noacq_marker_gate,ref_noacq_marker_acq,ref_noacq_marker_digitizer)
    ################################################################

    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms
    spectr.awg.ch12.load_waveform('Sat_pulse',
                              np.concatenate((I_sat, Q_sat)),
                              np.concatenate((sat_markersArray,sat_markersArray)))
    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_bsub',
                              np.concatenate((I_in_bsub,Q_in_bsub)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('refocused',
                              np.concatenate((I_refocused,Q_refocused)),
                              np.concatenate((refocused_markersArray,refocused_markersArray)))
    spectr.awg.ch12.load_waveform('ref_noacq',
                              np.concatenate((I_ref_noacq,Q_ref_noacq)),
                              np.concatenate((ref_noacq_markersArray,ref_noacq_markersArray)))
    spectr.awg.ch12.load_waveform('delay',
                              np.concatenate((delay_w1,delay_w2)),
                              np.concatenate((delay_markersArray,delay_markersArray)))
    spectr.awg.ch12.load_waveform('dummy',
                              np.concatenate((dummy_w1,dummy_w2)),
                              np.concatenate((dummy_markersArray,dummy_markersArray)))

    print('loaded in %.1fs'%(time()-start_time))

    return(acq_wait,end_delay)

def ramp_finished():
    if (i3d._instrument_x.ramping_state() == 'holding' and
        i3d._instrument_y.ramping_state() == 'holding' and
        i3d._instrument_z.ramping_state() == 'holding'):

        return True
    else:
        return False

def get_marker_and_field(field_axis=None,get='x'):
    marker_X = float(vna.ask('CALC:MARK1:X?'))
    marker_Y = float(vna.ask('CALC:MARK1:Y?').split(',')[0]) #Returns two values, second is irrelevant in most cases

    if not field_axis:
        field = float(i3d.field_measured()) # If axis not specified measure field magnitude
    elif field_axis=='x':
        field = float(i3d.x_measured())
    elif field_axis=='y':
        field = float(i3d.y_measured())
    elif field_axis=='z':
        field = float(i3d.z_measured())
    else:
        raise ValueError('Axis specified must be x, y, or z')

    if get=='x' or get=='X':
        marker=marker_X
    elif get=='x' or get=='Y':
        marker=marker_Y
    elif get=='both':
        marker=[marker_X,marker_Y]
    else:
        raise ValueError("'get' must be either 'x', 'y', or 'both'")

    field = [float(i3d.x_measured()),float(i3d.y_measured()),float(i3d.z_measured())]
    return (marker,field)

    marker_X = float(vna.ask('CALC:MARK1:X?'))
    marker_Y = float(vna.ask('CALC:MARK1:Y?').split(',')[0]) #Returns two values, second is irrelevant in most cases

    if not field_axis:
        field = float(i3d.field_measured()) # If axis not specified measure field magnitude
    elif field_axis=='x':
        field = float(i3d.x_measured())
    elif field_axis=='y':
        field = float(i3d.y_measured())
    elif field_axis=='z':
        field = float(i3d.z_measured())
    else:
        raise ValueError('Axis specified must be x, y, or z')

    if get=='x' or get=='X':
        marker=marker_X
    elif get=='x' or get=='Y':
        marker=marker_Y
    elif get=='both':
        marker=[marker_X,marker_Y]
    else:
        raise ValueError("'get' must be either 'x', 'y', or 'both'")

    field = [float(i3d.x_measured()),float(i3d.y_measured()),float(i3d.z_measured())]
    return (marker,field)

def BIP_silencing(Pi,pulse_amplitude,bip_Pi,bip_phases,bip_intervals,bip_amp,
                  wurst_Pi,wurst_freq,wurst_amp,long_wait, SR = 1.28e9/(2**5),
                  wait=50e-6, shot_rep=1,invert_type=None,refocus_type="wurst",
                  wurst_inversion_Pi=None,bip_inversion_Pi=None,inversion_wurst_amp = None,inversion_bip_amp = None,
                  bip_inversion_phases=None, bip_inversion_intervals=None,
                  phase = 0,python_avgs = 1,invert = False, pickle_input = True,pickle_output = True,
                  save= False, wait2 = None,name = "", folder = "C:\\Users\\Administrator\\Documents\\",
                  saveraw = False,N_inv = 1,d_inv = 150e-9,window = 12,refpulse_amp = 0, w=0):

    if refpulse_amp==0: refpulse_gate=0
    else: refpulse_gate = 1
    if wurst_inversion_Pi == None: wurst_inversion_Pi = wurst_Pi
    if inversion_wurst_amp == None: inversion_wurst_amp = wurst_amp
    if bip_inversion_Pi == None: bip_inersion_Pi = bip_Pi
    if inversion_bip_amp == None: inversion_bip_amp = bip_amp
    if bip_inversion_phases == None: bip_inversion_phases = bip_phases
    if bip_inversion_intervals == None: bip_inversion_intervals = bip_intervals
    if wait2==None: wait2 = 2*wait

    if invert == True:
        if invert_type != None:
            title2 = "With {} pre-inversion".format(invert_type)
            name = name+"{}_inverted".format(invert_type)

        else:
            raise Exception("Invert set to TRUE but no Type specified")

    else:title2 = "Without pre-inversion"


    print("Inversion type: {}\n".format(invert_type))
    print("Refocus type: {}\n".format(refocus_type))

    wfg.sample_rate = SR
    w = 0
    gate_wait = 50e-9
    frequency = sgs.frequency()
    current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5

    if spectr.dig.NumberOfAverages()==1:
        single_shot = 1  #This adds an extra digitiser trigger to make single shot take less time
        print("Single shot mode")
    else: single_shot=0

    ##############################################################
    #Define and load all fixed pulses and waveforms here:

    spectr.awg.ClearMemory()
    d1 = wfg.delay(150e-9,0)
    d1_inv = wfg.delay(d_inv,0)

    piebytwo = wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase)
    piebytwo_neg= wfg.heterodyne_pulse(pulse_amplitude,Pi/2,phase+180)

    d2 = wfg.heterodyne_delay(wait,0)

    pie = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase)
    pie_neg = wfg.heterodyne_pulse(pulse_amplitude,Pi,phase) # leave identical to pie unless you want to phase cycle the pi pulse!

    wurst = wfg.heterodyne_wurst_pulse(wurst_freq,phase,wurst_Pi,amp=wurst_amp)
    wurst_inversion = wfg.heterodyne_wurst_pulse(wurst_freq,phase,wurst_inversion_Pi,amp=inversion_wurst_amp)

    bip = wfg.BIP(bip_Pi,bip_phases,bip_intervals,bip_amp)
    bip_inversion = wfg.BIP(bip_inversion_Pi,bip_phases,bip_intervals,inversion_bip_amp)


    d3= wfg.heterodyne_delay(wait-19e-6,0)

    dt= wfg.pulse(single_shot,10e-6)
    dd = wfg.delay(spectr.RecordSize()/spectr.dig.SampleRate())

    print('loading...')

    ###############################################################
    #WURST pi pulse with acquisition

    wa_I,wa_Q     = wfg.heterodyne_combine(d1,wurst,d1,w = w,t0=0)

    wa_I_neg,wa_Q_neg = -wa_I,-wa_Q

    gate_len = len(wa_I)/SR
    acq_wait = gate_len+wait-25e-6

    wa_gate = wfg.pulse(1,gate_len)
    wa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    wa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    wa_pos = create_waveform("wa_pos",wa_I    , wa_Q    , wa_gate, wa_acq,wa_digitizer)
    wa_neg = create_waveform("wa_neg",wa_I_neg, wa_Q_neg, wa_gate, wa_acq,wa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #BIP pi with acquisition

    bipa_I,bipa_Q     = wfg.heterodyne_combine(d1,bip,d1,w = w,t0=0)

    bipa_I_neg,bipa_Q_neg = -bipa_I,-bipa_Q

    gate_len = len(bipa_I)/SR
    acq_wait = gate_len+wait-25e-6

    bipa_gate = wfg.pulse(1,gate_len)
    bipa_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    bipa_digitizer = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))

    bipa_pos = create_waveform("bipa_pos",bipa_I    , bipa_Q    , bipa_gate, bipa_acq,bipa_digitizer)
    bipa_neg = create_waveform("bipa_neg",bipa_I_neg, bipa_Q_neg, bipa_gate, bipa_acq,bipa_digitizer)
    #pickle_waveforms(*wa_pos[1:],SR)

    ###############################################################
    #WURST pulse

    wurst_I,wurst_Q     = wfg.heterodyne_combine(d1,wurst,d1,w = w,t0=0)

    wurst_I_neg,wurst_Q_neg = -wurst_I,-wurst_Q #wfg.heterodyne_combine(d1,-wurst_h,d1,w = w,t0=0)

    gate_len = len(wurst_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_pos = create_waveform("wurst_pos",wurst_I    , wurst_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_neg = create_waveform("wurst_neg",wurst_I_neg, wurst_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    ###############################################################
    #WURST inversion pulse

    wurst_inversion_I,wurst_inversion_Q     = wfg.heterodyne_combine(d1,wurst_inversion,d1_inv,w = w,t0=0)

    wurst_inversion_I_neg, wurst_inversion_Q_neg = -wurst_inversion_I,-wurst_inversion_Q

    #print(d_inv,len(wurst_inversion_I))

    gate_len = 150e-9-d_inv+len(wurst_inversion_I)/SR
    acq_wait = 0

    wurst_gate      = wfg.pulse(1,gate_len)
    wurst_acq       = wfg.delay(gate_len)
    wurst_digitizer = wfg.delay(gate_len)

    wurst_inversion_pos = create_waveform("wurst_inversion_pos",wurst_inversion_I    , wurst_inversion_Q    , wurst_gate, wurst_acq,wurst_digitizer)
    wurst_inversion_neg = create_waveform("wurst_inversion_neg",wurst_inversion_I_neg, wurst_inversion_Q_neg, wurst_gate, wurst_acq,wurst_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)

    ###############################################################
    #BIP

    bip_I,bip_Q     = wfg.heterodyne_combine(d1,bip,d1,w = w,t0=0)

    bip_I_neg,bip_Q_neg = -bip_I,-bip_Q #wfg.heterodyne_combine(d1,-wurst_h,d1,w = w,t0=0)

    gate_len = len(bip_I)/SR
    acq_wait = 0

    bip_gate      = wfg.pulse(1,gate_len)
    bip_acq       = wfg.delay(gate_len)
    bip_digitizer = wfg.delay(gate_len)

    bip_pos = create_waveform("bip_pos",bip_I    , bip_Q    , bip_gate, bip_acq,bip_digitizer)
    bip_neg = create_waveform("bip_neg",bip_I_neg, bip_Q_neg, bip_gate, bip_acq,bip_digitizer)

    ###############################################################
    #BIP inversion

    bip_inversion_I,bip_inversion_Q     = wfg.heterodyne_combine(d1,bip_inversion,d1_inv,w = w,t0=0)

    bip_inversion_I_neg, bip_inversion_Q_neg = -bip_inversion_I,-bip_inversion_Q

    #print(d_inv,len(wurst_inversion_I))

    gate_len = 150e-9-d_inv+len(bip_inversion_I)/SR
    acq_wait = 0

    bip_gate      = wfg.pulse(1,gate_len)
    bip_acq       = wfg.delay(gate_len)
    bip_digitizer = wfg.delay(gate_len)

    bip_inversion_pos = create_waveform("bip_inversion_pos",bip_inversion_I    , bip_inversion_Q    , bip_gate, bip_acq,bip_digitizer)
    bip_inversion_neg = create_waveform("bip_inversion_neg",bip_inversion_I_neg, bip_inversion_Q_neg, bip_gate, bip_acq,bip_digitizer)

    #pickle_waveforms(*wurst_pos[1:],SR)

    ###############################################################
    #Excitation waveform
    I_in_pos,Q_in_pos =  wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)
    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_neg,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,Pi/2+200e-9))
    marker_acq = wfg.delay(gate_len)
    marker_digitizer = wfg.delay(gate_len)

    In_pos = create_waveform("In_pos",I_in_pos, Q_in_pos, marker_gate,marker_acq,marker_digitizer)
    In_neg = create_waveform("In_neg",I_in_neg, Q_in_neg, marker_gate,marker_acq,marker_digitizer)

    #pickle_waveforms(*In_pos[1:],SR)

    ################################################################
    # delay waveform for reptime setting
    end_delay = 100e-6
    delay = create_waveform("delay",*[wfg.delay(end_delay)]*5)
    ###############################################################
    # delay waveform for inter-pulse delay
    short_delay = 1e-6
    ip_delay = create_waveform("ip_delay",*[wfg.delay(short_delay)]*5)
    ###############################################################
    # end trigger for digitiser reset
    r1 = wfg.delay(10e-6)
    r2 = wfg.pulse(refpulse_amp,10e-6)
    r3 = wfg.pulse(refpulse_gate,10e-6)
    reset = create_waveform("reset",r2,r1,r3,r1,wfg.combine(dd,dt))
    ###############################################################
    AWG_auto_setup(SR)

    ###############################################################
    ### WURST INVERSIONS:
    # wurst refocus
    SE_w_inv_w_refoc_p = ['SE_w_inv_w_refoc_p',[wurst_inversion_pos,delay,In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_w_inv_w_refoc_n = ['SE_w_inv_w_refoc_n',[wurst_inversion_pos,delay,In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])
    # bip refocus
    SE_w_inv_b_refoc_p = ['SE_w_inv_b_refoc_p',[wurst_inversion_pos,delay,In_pos,bip_pos,ip_delay,bipa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_w_inv_b_refoc_n = ['SE_w_inv_b_refoc_n',[wurst_inversion_pos,delay,In_neg,bip_pos,ip_delay,bipa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

    ### BIP INVERSIONS:
    # wurst refocus
    SE_b_inv_w_refoc_p = ['SE_b_inv_w_refoc_p',[bip_inversion_pos,delay,In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_b_inv_w_refoc_n = ['SE_b_inv_w_refoc_n',[bip_inversion_pos,delay,In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])
    # bip refocus
    SE_b_inv_b_refoc_p = ['SE_b_inv_b_refoc_p',[bip_inversion_pos,delay,In_pos,bip_pos,ip_delay,bipa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_p[0],[SE_inv_p[1][n][0] for n in range(len(SE_inv_p[1]))],SE_inv_p[2])

    SE_b_inv_b_refoc_n = ['SE_b_inv_b_refoc_n',[bip_inversion_pos,delay,In_neg,bip_pos,ip_delay,bipa_pos,reset,delay],
                    [N_inv,int(long_wait/end_delay),1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_inv_n[0],[SE_inv_n[1][n][0] for n in range(len(SE_inv_n[1]))],SE_inv_n[2])

    ### NO INVERSIONS:
    # wurst refocus:
    SE_n_inv_w_refoc_p = ['SE_n_inv_w_refoc_p',[In_pos,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n_inv_w_refoc_n = ['SE_n_inv_w_refoc_n',[In_neg,wurst_pos,ip_delay,wa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])
    # bip refocus:
    SE_n_inv_b_refoc_p = ['SE_n_inv_b_refoc_p',[In_pos,bip_pos,ip_delay,bipa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_p[0],[SE_p[1][n][0] for n in range(len(SE_p[1]))],SE_p[2])

    SE_n_inv_b_refoc_n = ['SE_n_inv_b_refoc_n',[In_neg,bip_pos,ip_delay,bipa_pos,reset,delay],
                    [1,1,int(wait2/short_delay),1,1,int(shot_rep/end_delay)]]
    spectr.awg.ch12.create_sequence(SE_n[0],[SE_n[1][n][0] for n in range(len(SE_n[1]))],SE_n[2])

    print('loaded')

    ###############################################################

    # Run sequences: positive and negative for phase cycling

    n_Is,n_Qs,p_Is,p_Qs = [],[],[],[]
    for n in range(python_avgs):
        start_time = time()

        if invert == True:
            if invert_type == "wurst" and refocus_type == "wurst":
                seq = SE_w_inv_w_refoc_n
            elif invert_type == "wurst" and refocus_type == "bip":
                seq = SE_w_inv_b_refoc_n
            elif invert_type == "bip" and refocus_type == "wurst":
                seq = SE_b_inv_w_refoc_n
            elif invert_type == "bip" and refocus_type == "bip":
                seq = SE_b_inv_b_refoc_n
        else:
            if refocus_type == "wurst":
                seq = SE_n_inv_w_refoc_n
            elif refocus_type == "bip":
                seq = SE_n_inv_b_refoc_n

        spectr.awg.ch12.init_channel(seq[0])
        n_I, n_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.3")
        n_Is.append(downsample(n_I,window))
        n_Qs.append(downsample(n_Q,window))
        n_I = np.mean(n_Is,axis = 0)
        n_Q = np.mean(n_Qs,axis = 0)

        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Negative phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))
        start_time = time()

        if invert == True:
            if invert_type == "wurst" and refocus_type == "wurst":
                seq = SE_w_inv_w_refoc_p
            elif invert_type == "wurst" and refocus_type == "bip":
                seq = SE_w_inv_b_refoc_p
            elif invert_type == "bip" and refocus_type == "wurst":
                seq = SE_b_inv_w_refoc_p
            elif invert_type == "bip" and refocus_type == "bip":
                seq = SE_b_inv_b_refoc_p
        else:
            if refocus_type == "wurst":
                seq = SE_n_inv_w_refoc_p
            elif refocus_type == "bip":
                seq = SE_n_inv_b_refoc_p

        spectr.awg.ch12.init_channel(seq[0])
        p_I, p_Q, timeI, timeQ = spectr.IQ_data_averaged()
        if pickle_input == True: pickle_sequence(seq,SR,address = "127.0.0.2")
        p_Is.append(downsample(p_I,window))
        p_Qs.append(downsample(p_Q,window))
        p_I = np.mean(p_Is,axis = 0)
        p_Q = np.mean(p_Qs,axis = 0)

        PhasedI = np.subtract(p_I, n_I)
        PhasedQ = np.subtract(p_Q, n_Q)

        #Demodulate from intermediate carrier frequency
        t_demod = np.array(np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait))
        I, Q = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

        #Downsample and average
        t = np.multiply(downsample(timeI,window),1e9)
        mag = np.sqrt(np.array(I)**2+np.array(Q)**2)

        #Plot
        if pickle_output == True: plot_pickle(t/1e9,I,Q)

        if save==True:
            if saveraw == True:
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pI.txt",p_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_pQ.txt",p_Qs)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nI.txt",n_Is)
                np.savetxt(folder+"\\"+name+"silencedEcho_raw_nQ.txt",n_Qs)
            np.savetxt(folder+"\\"+name+"silencedEcho.txt",np.transpose([t,I,Q,mag]))
            plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6), show = False)
            plt.savefig(folder+"\\"+name+"silencedEcho.pdf")
            plt.close()
        while (time()-start_time)<(shot_rep+python_delay-0.005): sleep(0.01)
        print('Positive phase average %i of %i completed in %.3f s'%(n+1,python_avgs,time()-start_time))

    plot_IQmag(t,I,Q,mag,title2 = title2,title1 = 'B = %.3fmT, f = %.3fMHz'%(current_field*1e3,frequency*1e-6))
    return(t,I,Q,mag)
