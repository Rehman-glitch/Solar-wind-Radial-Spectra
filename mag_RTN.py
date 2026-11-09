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
dt = np.median(np.diff(tg))
t = np.linspace(tg[0], tg[-1], int((tg[-1]-tg[0])/dt))

b_r = data.mag["B_R"].values        # radial component of magnetic field
b_t = data.mag["B_T"].values        # tangential component
b_n = data.mag["B_N"].values        # normal component
b_mag = data.mag["B_mag"].values    # total field strength (magnitude)

print(f"\nb_mag={b_mag}")
print(f"dt={dt}")

b = []
for coord in [b_r, b_t, b_n]:
    # Drop NaNs from both tg and coord
    valid = ~np.isnan(tg) & ~np.isnan(coord)  # ~ means not, this give a boolean array
    coord_interp = np.interp(t, tg[valid], coord[valid])
    b.append(coord_interp)

B, w = F.fast_fourier(b, t)
logw = np.log10(w)

b_trace = []                            # one trace per chunk
for i in range(len(B[0])):
    tr = 0                              # running sum of squares for chunk i
    for coord in B:
        tr += coord[i]**2
    b_trace.append(np.sqrt(tr))
B.append(b_trace)

dict = {
        # "B_R":B[0],
        # "B_T":B[1],
        # "B_N":B[2],
        "B_trace":B[3]
        }

fig, axs = mat.subplots(len(dict), 2, figsize=(10, 4), squeeze=False )

# figsize = (8,12) for all plots
# figsize = (10,4) for just trace

i=0
for key, value in dict.items():
    print(f"Plotting {key}")
    for j in range(len(value)):         # j-th chunk, one curve per chunk
        print(f"chunk {j+1}")
        coord = value[j]
        power = 2*dt*(coord)**2
        power = F.smoothen(w, power)
        logp = np.log10(power)
        
        m = F.regression_slope(logw, logp)
        
        axs[i, 0].plot(logw, logp, label=f'chunk={j+1}')
        axs[i, 1].plot(logw, m, label=f'chunk={j+1}')
    
    Davisy = [7, 6, 2]
    Davisx = [-4, -2, 0]
    
    axs[i, 0].plot(Davisx, Davisy)
    axs[i, 0].set_xlabel('log(f)')
    axs[i, 0].set_ylabel(f'log(power) [{key}]')
    axs[i, 0].set_title(f'{key} power spectrum')

    axs[i, 1].set_xlabel('log(f)')
    axs[i, 1].set_ylabel('spectral index (slope)')
    axs[i, 1].set_title(f'{key} slope')
    
    axs[i, 0].set_xlim(-4, 0)
    axs[i, 0].set_ylim(0, 8) 
    axs[i, 1].set_xlim(-4, 0)
    axs[i, 1].set_ylim(-2.5, 0)     
    
    axs[i, 0].legend()
    axs[i, 1].legend()
    i += 1

fig.tight_layout()
mat.show()
    

# Ad astra abyssosque!
