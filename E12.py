# E12

import sys
sys.path.append(r"C:\Users\ual-laptop\Desktop\Plasma Research\Codes") 
from solarwind import load_timeseries
import matplotlib.pyplot as mat
import numpy as np
import functions3 as F
import astropy.units as ap
import sunpy.coordinates as sp
from astropy.time import Time
import cdflib

k_B = 1.380649e-23   # Boltzmann constant, J/K
e = 1.602176634e-19    # proton charge, C
m_p = 1.67262192e-27   # proton mass, kg

# Bmag was in nT converted to T
# v was in km/s converted to m/s
# T was in eV converted to K

path = (r"C:\Users\ual-laptop\Desktop\Plasma Research\solo_L2_swa-pas-grnd-mom_20220601_V02.cdf")

cdf = cdflib.CDF(path)
# print(cdf.cdf_info())

t = Time('2022-06-01T00:00:00')

r_psp = sp.get_horizons_coord('Parker Solar Probe', t)
r_solo = sp.get_horizons_coord('Solar Orbiter', t)

r_psp = r_psp.radius.to(ap.au)
r_solo = r_solo.radius.to(ap.au)

# This converts the coordinates into distance

print('r_psp:', r_psp)
print('r_solo:', r_solo)

t_inj_psp = F.injection_time(r_psp.value)

print('psp injection time: ', t_inj_psp)

t_start_psp = t.iso
t_end_psp = (t + t_inj_psp * ap.s).iso

print('psp start: ', t_start_psp)
print('psp end: ', t_end_psp)

t_inj_solo = F.injection_time(r_solo.value)

print('solo injection time: ', t_inj_solo)

t_start_solo = t.iso
t_end_solo = (t + t_inj_solo * ap.s).iso

print('solo start: ', t_start_solo)
print('solo end: ', t_end_solo)

data_psp = load_timeseries(
    mission="psp",
    start=t_start_psp,
    stop=t_end_psp,
)

data_solo = load_timeseries(
    mission="SolO",
    start=t_start_solo,
    stop=t_end_solo,
)

'''
period = cdf.varget("Epoch")
t_cdf = cdflib.cdfepoch.to_datetime(period)
print('window:', t_start_solo, '->', t_end_solo)
print('file:  ', t_cdf[0], '->', t_cdf[-1])
'''

# ---------------- PSP ----------------

tg_psp = data_psp.mag['time'].values
tg_psp = (tg_psp - tg_psp[0]) / np.timedelta64(1, 's')  # The g stands for gaps in data
t_psp = np.linspace(tg_psp[0], tg_psp[-1], len(tg_psp))
dt_psp = (t_psp[-1]-t_psp[0])/(len(t_psp)-1)

b_r_psp = data_psp.mag["B_R"].values        # radial component of magnetic field
b_t_psp = data_psp.mag["B_T"].values        # tangential component
b_n_psp = data_psp.mag["B_N"].values        # normal component

b_mag_psp = data_psp.mag["B_mag"].values    # total field strength (magnitude)
bmag_psp = np.nanmedian(b_mag_psp)*1e-9

v_psp = data_psp.protons["V_R"].values      # velocities
v_psp = np.nanmedian(v_psp)*1e3

T_psp = data_psp.protons["T_p"].values        # temperatures
T_psp = np.nanmedian(T_psp)*11604.5

b_psp = []
for coord in [b_r_psp, b_t_psp, b_n_psp]:
    # Drop NaNs from both tg and coord
    valid = ~np.isnan(tg_psp) & ~np.isnan(coord)  # ~ means not, this give a boolean array
    coord_interp = np.interp(t_psp, tg_psp[valid], coord[valid])
    b_psp.append(coord_interp)

B_psp, w_psp = F.fast_fourier(b_psp, t_psp)

k_psp = w_psp/v_psp
vth_psp = np.sqrt((2*k_B*T_psp)/(m_p))
omega_psp = (e*bmag_psp)/m_p
k_psp = k_psp*(vth_psp/omega_psp)

logk_psp = np.log10(k_psp)

b_trace_psp = 0
for coord in B_psp:
    b_trace_psp += coord**2
b_trace_psp = np.sqrt(b_trace_psp)
B_psp.append(b_trace_psp)

dict_psp = {
        # "B_R":B_psp[0],
        # "B_T":B_psp[1],
        # "B_N":B_psp[2],
        "B_trace":B_psp[3]
        }


# ---------------- SolO ----------------

tg_solo = data_solo.mag['time'].values
tg_solo = (tg_solo - tg_solo[0]) / np.timedelta64(1, 's')  # The g stands for gaps in data
t_solo = np.linspace(tg_solo[0], tg_solo[-1], len(tg_solo))
dt_solo = (t_solo[-1]-t_solo[0])/(len(t_solo)-1)

b_r_solo = data_solo.mag["B_R"].values        # radial component of magnetic field
b_t_solo = data_solo.mag["B_T"].values        # tangential component
b_n_solo = data_solo.mag["B_N"].values        # normal component

b_mag_solo = data_solo.mag["B_mag"].values    # total field strength (magnitude)
bmag_solo = np.nanmedian(b_mag_solo)*1e-9

v_solo = data_psp.protons["V_R"].values      # Set to psp TEMPORARILY!!!
v_solo = np.nanmedian(v_solo)*1e3

T_solo = data_psp.protons["T_p"].values      # Set to psp TEMPORARILY!!!
T_solo = np.nanmedian(T_solo)*11604.5


b_solo = []
for coord in [b_r_solo, b_t_solo, b_n_solo]:
    # Drop NaNs from both tg and coord
    valid = ~np.isnan(tg_solo) & ~np.isnan(coord)  # ~ means not, this give a boolean array
    coord_interp = np.interp(t_solo, tg_solo[valid], coord[valid])
    b_solo.append(coord_interp)

B_solo, w_solo = F.fast_fourier(b_solo, t_solo)

k_solo = w_solo/v_solo
vth_solo = np.sqrt((2*k_B*T_solo)/(m_p))
omega_solo = (e*bmag_solo)/m_p
k_solo = k_solo*(vth_solo/omega_solo)
logk_solo = np.log10(k_solo)

b_trace_solo = 0
for coord in B_solo:
    b_trace_solo += coord**2
b_trace_solo = np.sqrt(b_trace_solo)
B_solo.append(b_trace_solo)

dict_solo = {
        # "B_R":B_solo[0],
        # "B_T":B_solo[1],
        # "B_N":B_solo[2],
        "B_trace":B_solo[3]
        }


print(v_psp, T_psp, bmag_psp, vth_psp, omega_psp)
print(v_solo, T_solo, bmag_solo, vth_solo, omega_solo)


# ---------------- PSP noise floor (digitized from figure) ----------------
# f in Hz, power in nT^2/Hz. 10 points each (straight lines in log-log between them).

f_mag = np.array([3.90e-02, 9.73e-02, 2.43e-01, 6.05e-01, 1.51e+00, 3.76e+00, 9.39e+00, 2.34e+01, 5.84e+01, 1.46e+02])
f_scm = np.array([1.10e+00, 2.00e+00, 4.00e+00, 1.00e+01, 3.00e+01, 1.00e+02, 3.00e+02, 8.00e+02, 2.50e+03, 9.50e+03])

p_mag_x = np.array([4.58e-04, 2.73e-04, 1.27e-04, 6.78e-05, 4.18e-05, 2.74e-05, 2.00e-05, 1.69e-05, 1.47e-05, 1.34e-05])
p_mag_y = np.array([8.82e-04, 3.20e-04, 1.05e-04, 5.10e-05, 2.82e-05, 1.74e-05, 1.45e-05, 1.21e-05, 1.06e-05, 9.19e-06])
p_mag_z = np.array([1.43e-03, 4.74e-04, 3.07e-04, 1.70e-04, 1.15e-04, 7.80e-05, 5.22e-05, 3.41e-05, 2.43e-05, 2.03e-05])

p_scm_x = np.array([1.06e-03, 1.45e-04, 2.03e-05, 4.11e-06, 4.64e-07, 3.91e-08, 5.22e-09, 1.14e-09, 3.51e-09, 3.73e-08])
p_scm_y = np.array([1.33e-03, 1.19e-04, 2.12e-05, 2.75e-06, 2.97e-07, 2.86e-08, 3.72e-09, 8.60e-10, 2.52e-09, 2.83e-08])
p_scm_z = np.array([1.44e-03, 8.93e-05, 1.69e-05, 1.74e-06, 2.09e-07, 1.95e-08, 2.46e-09, 4.96e-10, 1.51e-09, 1.82e-08])

# Trace noise = sum of the three component noise powers (same as B_trace^2)
p_mag_trace = p_mag_x + p_mag_y + p_mag_z
p_scm_trace = p_scm_x + p_scm_y + p_scm_z

# Convert noise frequency to the same x-axis as the PSP spectrum: k*rho = (2*pi*f/v)*(vth/omega)
logk_mag = np.log10((2*np.pi*f_mag/v_psp)*(vth_psp/omega_psp))
logk_scm = np.log10((2*np.pi*f_scm/v_psp)*(vth_psp/omega_psp))
logp_mag = np.log10(p_mag_trace)
logp_scm = np.log10(p_scm_trace)


# ---------------- Plots ----------------

fig, axs = mat.subplots(len(dict_psp), 2, figsize=(10, 4), squeeze=False )

# PSP and SolO overlaid on the same axes
# figsize = (8,12) for all plots
# figsize = (10,4) for just trace

i=0
for key in dict_psp:
    print(f"Plotting {key}")
    
    # PSP
    power_psp = 2*dt_psp*(dict_psp[key])**2
    power_psp = F.smoothen(w_psp, power_psp)
    logp_psp = np.log10(power_psp)
    m_psp = F.regression_slope(logk_psp, logp_psp)
    
    # SolO
    power_solo = 2*dt_solo*(dict_solo[key])**2
    power_solo = F.smoothen(w_solo, power_solo)
    logp_solo = np.log10(power_solo)
    m_solo = F.regression_slope(logk_solo, logp_solo)
    
    axs[i, 0].plot(logk_psp, logp_psp, label='PSP', color='green')
    axs[i, 0].plot(logk_solo, logp_solo, label='SolO', color='orange')
    
    # Noise Floor
    # axs[i, 0].plot(logk_mag, logp_mag, '--', label='PSP MAG noise', color='gray')
    # axs[i, 0].plot(logk_scm, logp_scm, ':', label='PSP SCM noise', color='black')
    
    axs[i, 0].set_xlabel('log(k)')
    axs[i, 0].set_ylabel(f'log(power) [{key}]')
    axs[i, 0].set_title(f'{key} power spectrum')
    axs[i, 0].legend()

    axs[i, 1].plot(logk_psp, m_psp, label='PSP', color='green')
    axs[i, 1].plot(logk_solo, m_solo, label='SolO', color='orange')
    axs[i, 1].set_xlabel('log(k)')
    axs[i, 1].set_ylabel('spectral index (slope)')
    axs[i, 1].set_title(f'{key} slope')
    axs[i, 1].legend()
    
    '''
    axs[i, 0].set_xlim(-4, 1)
    axs[i, 0].set_ylim(-6, 6) 
    axs[i, 1].set_xlim(-4, 1)
    axs[i, 1].set_ylim(-4, 0)
    '''
    
    i += 1
    
fig.tight_layout()
#fig.savefig('E12 PSD against wavenumber.pdf', bbox_inches='tight')
mat.show()
