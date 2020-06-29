import lmfit
from lmfit.models import BreitWignerModel,LinearModel, QuadraticModel
import socket
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import leastsq
from scipy.optimize import least_squares

def basic_fit_FanoResonance(freq,trace,filename = 'untitled', plot = True,save = True):
    start,stop = None, None #np.argmax(trace)-500,np.argmax(trace)+500# 27900,28200  #Specifies the window within the data to analyse. Set to None,None if you want the whole window
    Lin_mod = LinearModel()                                         #Linear lmfit model for background offset and slope
    BW_mod = BreitWignerModel()                                     #Breit-Wigner-Fano model
    mod = BW_mod+Lin_mod
    x = freq[start:stop]/1E6                                        #Convert frequencies to MHz
    trace = (10**(trace/10))                                        #Convert decibel data to linear
    y = trace[start:stop]
    pars = BW_mod.guess(y, x=x)                                     #Initialize fit params
    pars += Lin_mod.guess(y,x=x, slope = 0, vary = False)           
    pars['center'].set(value=x[np.argmax(y)], vary=True, expr='')   #Use numpy to find the highest transmission value. Corresponding frequency is used as a guess for the centre frequency
    pars['sigma'].set(value=0.1, vary=True, expr='')                #Linewidth
    pars['q'].set(value=1, vary=True, expr='')                      #Fano factor (asymmetry term). q=infinite gives a Lorentzian
    pars['amplitude'].set(value=-0.03, vary=True, expr='')          #Amplitude
    out  = mod.fit(y,pars,x=x)
    sigma = out.params['sigma']
    centre = out.params['center']
    return(x,y,out.best_fit,sigma.value,centre.value,centre.value/sigma.value)       #Returns linewidth in GHz, centre in GHz and Q factor

def plot_pickle(x,y1,y2):
    #erver_address = ('localhost', 10000)
    server_address = ("127.0.0.1", 10000)
    sock = socket.socket()
    sock.connect(server_address)
    serialized_data = pickle.dumps([x, y1,x,y2], protocol=2)
    sock.sendall(serialized_data)
    sock.close()
    
def quadratic_fit(x0,y0,start = 0, stop = None,colour = "#23c161"):
    x = x0[start:stop]
    y = y0[start:stop]
    quad_mod = QuadraticModel()
    pars = quad_mod.guess(y, x=x)
    out  = quad_mod.fit(y,pars,x=x)
    a,b,c = out.params["a"],out.params["b"],out.params["c"]
    x_fit = np.linspace(min(x),max(x),100)
    y_fit = a*x_fit**2+b*x_fit+c
    x0_fit = np.linspace(min(x0),max(x0),100)
    y0_fit = a*x0_fit**2+b*x0_fit+c
    return(a.value,b.value,c.value)

def quadratic_fn(x, a, b, c):
    return (a*x**2 + b*x +c)

def downsample_simple(x,window):
    return(np.take(x,np.linspace(0,len(x)-1,int(len(x)/window),dtype = int)))
    
def downsample(data,window):
    averaged = []
    for n in range(int(len(data)/window)):
        averaged.append(np.mean(data[n*window:(n+1)*window]))
    return(averaged)
    
def plot_IQ(t,I,Q,title = "",show = True):
    t = np.array(t)
    plt.plot(t/1000,I,label = "I")
    plt.plot(t/1000,Q,label = "Q")
    plt.title(title)
    plt.xlabel(u"Time (\u03bcs)")
    plt.ylabel("Signal (V)")
    plt.legend()
    if show == True: plt.show()
  
def plot_mag(t,mag,title = "",show = True):
    t = np.array(t)
    plt.plot(t/1000,mag, label = "magnitude", color = "Indianred")
    plt.title(title)
    plt.legend()
    plt.xlabel(u"Time (\u03bcs)")
    plt.ylabel("Signal magnitude (V)")
    if show == True: plt.show()

def plot_IQmag(t,I,Q,mag,title1 = '',title2 = '',show = True):
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plot_IQ(t,I,Q,title = title1,show = False)
    plt.subplot(1, 2, 2)
    
    plot_mag(t,mag,title = title2,show = False)
    if show == True: plt.show()

def fit_function(guess,func,xdata,ydata,lb = -np.inf,ub = np.inf):
    
    optimize_func = lambda x: func(x,xdata)- ydata
    
    # we'll use this to plot our first estimate
    data_first_guess = func(guess,xdata)

    full_est = least_squares(optimize_func, guess,bounds = (lb,ub))
    est = full_est.x
    
    J = full_est.jac
    cov = np.linalg.inv(J.T.dot(J))
    var = np.sqrt(np.diagonal(cov))
    print (est)
    print (var)
    # recreate the fitted curve using the optimized parameters
    fine = np.linspace(min(xdata),max(xdata),len(xdata)*10)
    data_fit=func(est,fine)
    return (est,fine,data_fit)

def T2_fit(mag_int,t,noise_floor = 0,print_T2 = True,plot = True,title = None, show = True,save=False,filename = 'T2.pdf'):
    T2_func = lambda y,x: ((y[0]*np.exp(-x/y[1]))**2+noise_floor**2)**0.5
    est,fine,data_fit = fit_function([0.1,100e-6],T2_func,t,mag_int)
    if print_T2 == True: print (u"T2 = %.3f \u03bcs"%(est[1]*2e6))
        
    if plot == True:    
        #print("hello, I am resonator tools")
        plt.plot(t*1e6,mag_int,'o')
        plt.plot(fine*1e6,data_fit,label = u"T$_2$ = %.3f \u03bcs"%(est[1]*2e6))
        plt.title(title)
        plt.xlabel(u"Time (\u03bcs)")
        plt.ylabel("Integrated echo signal(V)")
        plt.tight_layout()
        plt.xlim(0,1e6*t.max())
        plt.legend()
        #plt.yscale('log')
    if save == True: plt.savefig(filename)
    if show == True: plt.show()
        
    return est[1]
    
def T1Plot(I_int,Q_int,x):
    YI = []
    YQ = []
    
    N = len(I_int)
    t =  x*1e3
    data = np.multiply(Q_int,100)
    
    
    guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
    guess_zero = np.mean(data[-1:])
    guess_amp = 0.5
    guess_T1 =1
    
    # we'll use this to plot our first estimate
    data_first_guess = guess_amp*np.exp(-t/guess_T1)+guess_zero
    
    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.exp(-t/x[1])+guess_zero - data
    est_amp, est_T1 = leastsq(optimize_func, [guess_amp,guess_T1])[0]
    
    
    
    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*np.exp(-t/est_T1)+guess_zero
    
    # recreate the fitted curve using the optimized parameters
    fine_t = np.arange(0,max(t),1)
    data_fit=est_amp*np.exp(-fine_t/est_T1)+guess_zero
    
    print ("T1 = %.ims"%(est_T1))
    
    plt.plot(t, data/10, 'o')
    
    #plt.plot(t*1e3, data_first_guess/10, label='first guess')
    plt.plot(fine_t, data_fit/10, label=("T$_1$ = %.1f ms"%(est_T1)))
    plt.legend()
    plt.ylabel('Echo Magnitude (arb.unit)')
    plt.xlabel('Delay (ms)')
    plt.savefig(name+'_%.2famp_T1.pdf'%pulse_amplitude)
    
def integrate_2dsweep(Is,Qs,start,stop,sweep1,sweep2):
    I = []
    Q = []
    mag = []
    for n in range(len(sweep1)):
        In =np.array(Is[n*len(sweep2):(n+1)*len(sweep2)])
        Qn = np.array(Qs[n*len(sweep2):(n+1)*len(sweep2)])
        I.append(In)
        Q.append(Qn)
        mag.append((In**2+Qn**2)**0.5)
        #print(np.mean((In**2+Qn**2)**0.5))
    
    #print(np.shape(I[0]))
    I_int = []
    Q_int = []
    mag_int = []

    for n in range(len(sweep1)):
        I_int_0 = []
        Q_int_0 = []
        mag_int_0 = []
        for t in range(len(I[n])):
            I_int_0.append(np.mean(I[n][t][start:stop]))
            Q_int_0.append(np.mean(Q[n][t][start:stop]))
            mag_int_0.append(np.mean(mag[n][t][start:stop]))
        I_int.append(I_int_0)
        Q_int.append(Q_int_0)
        mag_int.append(mag_int_0)
        
    return(I_int,Q_int,mag_int)

def plot_2d_sweep(data,x=None,y=None,xlabel = '',ylabel = '',clabel = '',title = '',xtick = 1,ytick = 1,centre = 0,vmin = None,vmax = None,cmap = sns.diverging_palette(240, 10, n=361)):
    #if x!=None and y!=None: 
    fieldsweep_df = pd.DataFrame(data=np.flip(data,axis = 0),index=np.flip(y,axis = 0),columns=x)
    #else: fieldsweep_df = pd.DataFrame(data=data)
    ax = sns.heatmap(fieldsweep_df, xticklabels = xtick, yticklabels = ytick,cmap = cmap,center = centre,vmin = vmin, vmax = vmax)#,center = -100,cmap = sns.diverging_palette(240, 10, n=361))
    ax.collections[0].colorbar.set_label(clabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
def fieldvec(current_field,r01,theta01,phi01,n):
    r = 0.001*np.linspace(r01[0],r01[1],n)
    theta = np.linspace(theta01[0],theta01[1],n)# (-2,-2,n)#
    phi = np.linspace(phi01[0],phi01[1],n)
    fields = list(zip(r, theta, phi))
    
    step0 = abs(fields[0][0]- np.linalg.norm(np.array(current_field)))
    if (step0 > 0.0201): raise Exception('Abort ramp: first step %.3fmT is larger than 20mT'%(1000*step0))
    
    if n>1:
        stepr = abs(r[1]-r[0])
        steptheta = abs(theta[1]-theta[0])
        stepphi = abs(phi[1]-phi[0])
        if (stepr > 0.0201): raise Exception('Abort ramp: step %.3fmT is larger than 20mT'%(1000*stepr))
        if stepr > 0 and (steptheta+stepphi>0): raise Exception('Ramping two values at once')
        elif steptheta > 0 and (stepr+stepphi>0): raise Exception('Ramping two values at once')
        
    return(fields)
    
def rotate(I,Q,theta):
    theta = theta*np.pi/180
    
    I_prime = I*np.cos(theta)+Q*np.sin(theta)
    Q_prime = Q*np.cos(theta)-I*np.sin(theta)
    return(I_prime,Q_prime)    

def T2_stretchedfit(mag_int,t,noise_floor = 0,print_T2 = True,plot = True,title = None, show = True,save=False,filename = 'T2.pdf'):
    T2_func = lambda y,x: ((y[0]*np.exp(-(x/y[1])**y[2]))**2+noise_floor**2)**0.5
    est,fine,data_fit = fit_function([max(mag_int),400e-6,2],T2_func,t,mag_int)
    if print_T2 == True: print (u"T2 = %.3f \u03bcs"%(est[1]*2e6))
        
    if plot == True:    
        plt.plot(t*1e6,mag_int,'o')
        plt.plot(fine*1e6,data_fit,label = u"T$_2$ = %.3f \u03bcs\nn = %.2f"%(est[1]*2e6,est[2]))
        plt.title(title)
        plt.xlabel(u"Time (\u03bcs)")
        plt.ylabel("Integrated signal(V)")
        plt.tight_layout()
        plt.xlim(0,1e6*t.max())
        plt.legend()
        #plt.yscale('log')
    if save == True: plt.savefig(filename)
    if show == True: plt.show()
        
    return est[1],est[2]

def T2_biexpfit(mag_int,t,noise_floor = 0,print_T2 = True,plot = True,title = None, show = True,save=False,filename = 'T2.pdf'):
    T2_func = lambda y,x: ((y[0]*np.exp(-(x/y[1])+y[3]*np.exp(-x/y[2])))**2+noise_floor**2)**0.5
    est,fine,data_fit = fit_function([0.1,100e-6,10e-6,1],T2_func,t,mag_int)
    if print_T2 == True: print (u"T2 = %.3f \u03bcs"%(est[1]*2e6))
        
    if plot == True:    
        plt.plot(t*1e6,mag_int,'o')
        plt.plot(fine*1e6,data_fit,label = u"$T_2$ = %.3f \u03bcs"%(est[1]*2e6))
        plt.title(title)
        plt.xlabel(u"Time (\u03bcs)")
        plt.ylabel("Integrated echo signal(V)")
        plt.tight_layout()
        plt.xlim(0,1e6*t.max())
        plt.legend()
        #plt.yscale('log')
    if save == True: plt.savefig(filename)
    if show == True: plt.show()
        
    return est[1]
