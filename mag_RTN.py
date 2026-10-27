from solarwind import load_timeseries
import matplotlib.pyplot as mat
import numpy as np


data = load_timeseries(
    mission="psp",
    start="2021-11-17T07:00:00",
    stop="2021-11-20T06:00:00",
)

tg = data.mag['time'].values
tg = (tg - tg[0]) / np.timedelta64(1, 's')  # The g stands for gaps in data
t = np.linspace(tg[0], tg[-1], len(tg))
dt = (t[-1]-t[0])/(len(t)-1)

b_r = data.mag["B_R"].values        # radial component of magnetic field
b_t = data.mag["B_T"].values        # tangential component
b_n = data.mag["B_N"].values        # normal component
b_mag = data.mag["B_mag"].values    # total field strength (magnitude)

b = []
for coord in [b_r, b_t, b_n]:
    # Drop NaNs from both tg and coord
    valid = ~np.isnan(tg) & ~np.isnan(coord)  # ~ means not, this give a boolean array
    coord_interp = np.interp(t, tg[valid], coord[valid])
    b.append(coord_interp)




def fast_fourier(y, x):
    
    dx = (x[-1]-x[0])/(len(x)-1)
    w = np.fft.rfftfreq(len(x), dx)
    w = w[1:]
    
    B = []
    for coord in y:
        b = np.fft.rfft(coord-np.mean(coord), norm="ortho")
        B.append(np.abs(b[1:]))
    
    return (B, w)

def smoothen(x, factor):
    c = np.append(0, np.cumsum(x))     
    bin = np.linspace(1, len(x), len(x))
    bin = factor*bin
    X = []
    for i in range(len(bin)):
        n = int(bin[i])
        m = i-(n//2)
        if m < 0:
            m = 0
        e = i+(n//2)+1
        if e > len(x):
            e = len(x)
        mean = (c[e]-c[m])/(e-m)
        X.append(mean)
    return X



B, w = fast_fourier(b, t)
w = smoothen(w,0.1)
logw = np.log10(w)

b_trace = 0
for coord in B:
    b_trace += coord**2
b_trace = np.sqrt(b_trace)
B.append(b_trace)

fig, axs = mat.subplots(len(B), 1, figsize=(8, 10))

for i in range(len(B)):
    coord = B[i]
    power = dt*(coord)**2
    power = smoothen(power,2)
    logp = np.log10(power)
    
    axs[i].plot(logw, logp)
    
fig.tight_layout()
mat.show()
    
