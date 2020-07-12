import numpy as np
import matplotlib.pyplot as plt
from repeatedinteractions import Reservoir, System
from qutip import *
#------------Tiempo--------------
tf = 10
dt = 0.001
N = int(tf/dt)
td = 0.85
Nd = int(td / dt)
t = np.linspace(0,tf,N)
#-----------Hamiltoniano---------
h = 1.5
H = h/2 * sigmaz()
H_constant = []
for i in range(N):
    H_constant.append(H)
eps = np.sqrt(5)
V = eps * (tensor(sigmap(), sigmap()) + tensor(sigmam(), sigmam()))
#--------Condiciones iniciales------
bath = Reservoir(H, 1)
qubit = System(H_constant, 2)
qubit.state.append(bath.Thermal_state())
qubit.energy.append((qubit.state[0]*qubit.hamiltonian[0]).tr())
qubit.heat.append(0)
qubit.work.append(0)
#--------Evolución temporal--------
qubit.Time_evolution(bath.Thermal_state(), bath.hamiltonian, tf, dt, V)
qubit.energy[0] = 0
qubit.work[1] = 0
#---------Graficos--------------
t = np.linspace(0, qubit.stroke_count * N, len(qubit.energy))
plt.figure()
plt.plot(t, qubit.energy, 'C0-.', linewidth = 2)
plt.plot(t, qubit.heat,'r-.', linewidth = 2)
plt.plot(t, qubit.work, 'C7-.', linewidth = 2)
plt.show()