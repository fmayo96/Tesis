import numpy as np 
import matplotlib.pyplot as plt 

h_hot = 1.5
beta_hot = 1.0
beta_cold = 1.0
h_init = 1.5
omega = h_init / 2.0
delta = 0.001
hrange = np.arange(-h_init, h_init, delta)
tf = 20
dt = 0.001
N_time = int(tf/dt)
trange = np.arange(0, tf, dt)
h , t = np.meshgrid(hrange, trange)
w = np.sqrt(omega**2 - (h/2.0)**2)
Z_hot = 2*np.cosh(beta_hot*h_hot/2.0)
Z_cold = 2*np.cosh(beta_cold*h_init/2.0)
eps = np.sqrt(0.5)
x =  ((4*h**2 + eps**4)*np.exp(beta_cold*h_init/2.0)/Z_cold - 8*w**2) / (16*omega**2 + eps**4)
A = (np.exp(-beta_hot*h_hot/2.0)/Z_hot - np.exp(beta_cold*h_init/2.0)/Z_cold)
rho_00 = (1/omega**2)* A * np.exp(-eps*eps*t/2.0)*(h*h/4.0 + h*w/4.0*np.cos(2*omega*t)) + x 
rho_01 = (w/2*omega**2)*A* np.exp(-eps*eps*t/2.0) * (h*np.sin(omega*t)**2 + 1j*omega*np.sin(2*omega*t)) + (2j*w*(1 + 2*x)) / (eps**2 + 2j*h)
Energy = h * (rho_00 - 0.5) + 2*w*np.real(rho_01)


plt.figure()
plt.contourf(t,h,Energy, levels=50, cmap='jet')
plt.xlabel('Time', fontsize = 12)
plt.ylabel('h', fontsize = 12)
plt.title("Energy", fontsize = 12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.colorbar()
plt.show()