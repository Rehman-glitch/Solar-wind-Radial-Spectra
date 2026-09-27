import matplotlib.pyplot as mat
import numpy as np
from scipy import integrate

x = np.linspace(-50, 50, 1000)
y = np.sin(2*np.pi*x)

def fourier_transform(x, y, integral="trapezoid"):
    X = x[-1] - x[0]
    omega_min = 2*np.pi / X
    dx = X / len(x)
    omega_max = np.pi / dx
    # Just setting up the limits of omega.
    
    omegas = np.linspace(-omega_max, omega_max, 5000)
    
    F = []
    if integral == "simpson":
        for w in omegas:
            integrand = y * np.exp(-1j*w*x)
            F.append(integrate.simpson(integrand, x))
    else:
        for w in omegas:
            integrand = y * np.exp(-1j*w*x)
            F.append(np.trapezoid(integrand, x))
    
    F = np.array(F)
    
    return (F, omegas)

def fast_fourier(l, t):
    dt = (t[-1]-t[0])/len(t)
    w = 2*np.pi*np.fft.fftfreq(len(t), dt)
    w = np.fft.fftshift(w)
    transformed_array = []
    for axis in l:
        f = np.fft.fft(axis)
        f = np.fft.fftshift(f)
        transformed_array.append(f)
        
    transformed_array = np.array(transformed_array)
    
    return (transformed_array, w)

dt = x[1]-x[0]
f = np.fft.fft(y)
w = 2*np.pi*np.fft.fftfreq(len(x), dt)
f = np.fft.fftshift(f)
w = np.fft.fftshift(w)


mat.figure()
mat.plot(w,abs(f))
mat.show()
