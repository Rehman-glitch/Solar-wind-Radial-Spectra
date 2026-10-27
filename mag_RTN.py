import sys
sys.path.append(r"C:\Users\ual-laptop\Desktop\Plasma Research\Codes") 
from solarwind import load_timeseries
import matplotlib.pyplot as mat
import numpy as np
import functions2 as F

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

B, w = F.fast_fourier(b, t)
w = F.smoothen(w,0.5)
logw = np.log10(w)

b_trace = []                            # one trace per chunk
for i in range(len(B[0])):
    tr = 0                              # running sum of squares for chunk i
    for coord in B:
        tr += coord[i]**2
    b_trace.append(np.sqrt(tr))
B.append(b_trace)

dict = {
        "B_R":B[0],
        "B_T":B[1],
        "B_N":B[2],
        "B_trace":B[3]
        }

fig, axs = mat.subplots(len(B), 2, figsize=(10, 10))

i=0
for key, value in dict.items():
    
    for j in range(len(value)):         # j-th chunk, one curve per chunk
        coord = value[j]
        power = dt*(coord)**2
        power = F.smoothen(power,0.5)
        logp = np.log10(power)
        
        m = F.regression_slope(logw, logp, 1)
        
        axs[i, 0].plot(logw, logp)
        axs[i, 1].plot(logw, m)
    
    Davisy = [7, 6, 2]
    Davisx = [-4, -2, 0]
    
    axs[i, 0].set_xlabel('log(f)')
    axs[i, 0].set_ylabel(f'log(power) [{key}]')
    axs[i, 0].set_title(f'{key} power spectrum')

    axs[i, 1].set_xlabel('log(f)')
    axs[i, 1].set_ylabel('spectral index (slope)')
    axs[i, 1].set_title(f'{key} slope')
    
    i += 1
    
fig.tight_layout()
mat.show()
    
