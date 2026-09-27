from solarwind import load_timeseries
import matplotlib.pyplot as mat
import numpy as np
import functions as F

data = load_timeseries(
    mission="psp",
    start="2021-11-20T03:53:25",
    stop="2021-11-20T12:55:34",
)


# start="2022-02-25T00:00:00",
# stop="2022-02-26T00:00:00"

t = data.mag['time'].values
B = data.mag['B_mag'].values
t = (t - t[0]) / np.timedelta64(1, 's')


l = ["B_R", "B_T", "B_N"]

f, w = F.fast_fourier(l, "time")
B_sqrd = np.abs(f[0])**2 + np.abs(f[1])**2 + np.abs(f[2])**2
l.append("B_sqrd")
f = list(f)
f.append(B_sqrd)
f = np.array(f)


n = len(w)   # length of your w array
print(n)
step = n // 50000     # pick step so output ≈ 50000 points

omegas = F.smoothen(w, 10, step)

fig, axs = mat.subplots(len(l), 2, figsize=(10, 4*len(l)))

N = len(t)

for i in range(len(l)):
    axis = l[i]
    if axis == "B_sqrd":
        power = abs(f[i]) / N
    else:
        power = np.abs(f[i])**2 / N 
    power = F.smoothen(power, 10, step)
    logpower = np.log10(power)
    logomegas = np.log10(omegas)

    modx, gradients = F.regression_slope(logomegas, logpower, 5000, 100)
    # modx = smoothen(modx, 2, 2)
    # gradients = smoothen(gradients, 2, 2)
    
    axs[i, 0].plot(logomegas, logpower)
    axs[i, 0].set_xlabel('log(f)')
    axs[i, 0].set_ylabel(f'log(power) [{axis}]')
    axs[i, 0].set_title(f'{axis} power spectrum')

    # Right column: spectral index (regression slope) vs log(ω)
    axs[i, 1].plot(modx, gradients)
    axs[i, 1].set_xlabel('log(f)')
    axs[i, 1].set_ylabel('spectral index (slope)')
    axs[i, 1].set_title(f'{axis} slope')

fig.tight_layout()
mat.show()


#['SolarWindData', '__all__', '__builtins__', '__cached__', '__doc__', 
# '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 
# 'cache', 'data', 'load', 'load_timeseries']

'''
b_r = data.mag["B_R"]        # radial component of magnetic field
b_t = data.mag["B_T"]        # tangential component
b_n = data.mag["B_N"]        # normal component
b_mag = data.mag["B_mag"]    # total field strength (magnitude)

v_r = data.protons["V_R"]    # radial proton velocity
v_mag = data.protons["V_mag"]  # total proton speed
n_p = data.protons["n_p"]    # proton density
t_p = data.protons["T_p"]    # proton temperature
r_au = data.protons["r_au"]  # distance from Sun, in AU
'''

