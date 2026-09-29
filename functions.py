from solarwind import load_timeseries
import numpy as np

data = load_timeseries(
    mission="psp",
    start="2021-11-20T03:53:25",
    stop="2021-11-20T12:55:34",
)

def fourier_transform(l, time):
    '''
    l is the list of data you want tranformed
    t is the time interval
    
    Returns a tuple of the transformed lists and the omegas.
    '''
    
    t = data.mag[time].values
    t = (t - t[0]) / np.timedelta64(1, 's')  # seconds since start
    
    T = t[-1] - t[0]
    omega_min = 2*np.pi / T
    dt = T / len(t)
    omega_max = 1 / 2*dt
    # Just setting up the limits of omega.
    
    omegas = np.logspace(np.log10(omega_min), np.log10(omega_max), 200)
    
    dl = []
    for axis in l:
        B = data.mag[axis].values
        
        F = []
        # A single axis fourier tranformed
        for w in omegas:
            integrand = B * np.exp(-1j*w*t)
            F.append(np.trapezoid(integrand, t))
            print(str(w) + axis) 
        
        F = np.array(F)
        dl.append(F)

    tranformed_arrays = np.array(dl)
    
    return (tranformed_arrays, omegas)

def smoothen(l, r, step):
    '''
    l is the array to be smoothened
    r is the range over which mean is calculated
    step is how far apart are the data taken to be averaged over
    Returns a tuple of the smoothened (averaged) l and t
    '''
    
    ls = np.array([])
    
    for i in range(0, len(l)-r+1, step):
        
        ls_t = np.array([])
              
        for j in range(r):
            ls_t = np.append(ls_t, l[i+j])
            
        l_mean = np.mean(ls_t)
        ls = np.append(ls, l_mean)
        
    return ls    

def regression_slope(x, y, r, step):
    '''
    
    This function divides a data set into small section
    then draws a regression line through the small sections.
    and returns the slopes of the regression line as an array.
    
    Parameters
    ----------
    x : array of x values
    y : array of y values
    r : integer
        range to find slope over
    step : TYPE
        DESCRIPTION.

    Returns
    -------
    array of regression line slope
    
    Formulae
    --------
    
    Least squares regression for y = a + b*x

    b = Sxy / Sxx
    a = ybar - b * xbar

where:
    Sxy = sum(x*y) - (sum(x) * sum(y)) / n
    Sxx = sum(x**2) - (sum(x))**2 / n
    xbar = mean(x), ybar = mean(y), n = number of points

    '''
    
    gradients = []
    modx = []
    
    for i in range(0, len(x)-r+1, step):
        lx = x[i:i+r]
        ly = y[i:i+r]
        avgx = np.sum(x[i:i+step])/step
        # average of x over the interval
        Sxx = np.sum(lx**2) - (np.sum(lx)**2)/len(lx)
        Sxy = np.sum(lx*ly) - (np.sum(lx)*np.sum(ly))/len(lx)
        slope = Sxy/Sxx
        gradients.append(slope)
        modx.append(avgx)
        
    return (np.array(modx), np.array(gradients))

# Keep r and step same for best results.

def fast_fourier(l, time):
    
    t = data.mag[time].values
    t = (t - t[0]) / np.timedelta64(1, 's')
    dt = (t[-1]-t[0])/len(t)
    w =np.fft.rfftfreq(len(t), dt)
    
    print(dt)
    fw = 1/(t[-1]-t[0])
    print(fw)
    
    transformed_array = []
    for axis in l:
        B = data.mag[axis].values
        f = np.fft.rfft(B, norm="forward")
        transformed_array.append(f)
        print(axis)
        
    transformed_array = np.array(transformed_array)
    
    return (transformed_array, w)
