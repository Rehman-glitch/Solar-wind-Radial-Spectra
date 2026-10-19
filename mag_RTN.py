import sys
sys.path.append(r"C:\Users\ual-laptop\Desktop\Plasma Research\Codes") 
from solarwind import load_timeseries
import matplotlib.pyplot as mat
import numpy as np
import functions as F

data = load_timeseries(
    mission="psp",
    start="2021-11-17T07:00:00",
    stop="2021-11-20T08:00:00",
)

t = data.mag['time'].values
B = data.mag['B_mag'].values
t = (t - t[0]) / np.timedelta64(1, 's')
dt = (t[-1]-t[0])/len(t)

for a in ["B_R", "B_T", "B_N"]:
    print(a, int(data.mag[a].isnull().sum()))

l = ["B_R", "B_T", "B_N"]

f, w = F.fast_fourier(l, "time")
B_trace = np.abs(f[0])**2 + np.abs(f[1])**2 + np.abs(f[2])**2
l.append("B_trace")
f = list(f)
f.append(B_trace)
f = np.array(f)

n = len(w)   # length of your w array
print("Length of w array {}".format(n))
step = n // 5000     # pick step so output ≈ 50000 points

omegas = F.smoothen(w, 10, step)
logomegas = np.log10(omegas)

fig, axs = mat.subplots(len(l), 2, figsize=(10, 4*len(l)))

N = len(t)

for i in range(len(l)):
    axis = l[i]

    for k in range(len(f[i])):                  # one curve per chunk
        if axis == "B_trace":
            power = dt*abs(f[i][k])
        else:
            power = dt*np.abs(f[i][k])**2
        power = F.smoothen(power, 10, step)
        logpower = np.log10(power)

        modx, gradients = F.regression_slope(logomegas, logpower, 5000, 100)

        axs[i, 0].plot(logomegas, logpower, label=f"chunk {k+1}")
        axs[i, 1].plot(modx, gradients)

    Davisy = [7, 6, 2]
    Davisx = [-4, -2, 0]
    axs[i, 0].plot(Davisx, Davisy, color="k", label="Davis")
    axs[i, 0].set_xlabel('log(f)')
    axs[i, 0].set_ylabel(f'log(power) [{axis}]')
    axs[i, 0].set_title(f'{axis} power spectrum')

    axs[i, 1].set_xlabel('log(f)')
    axs[i, 1].set_ylabel('spectral index (slope)')
    axs[i, 1].set_title(f'{axis} slope')

axs[0, 0].legend()
fig.tight_layout()
mat.show()
