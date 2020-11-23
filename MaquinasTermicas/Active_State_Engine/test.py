import numpy as np
import matplotlib.pyplot as plt 
from repeatedinteractions_c import Reservoir, System
from pauli import pauli
#------------Time--------------
tf = 15
dt = 0.001
N = int(tf/dt)
td = 0.85
Nd = int(td / dt)
t = np.linspace(0,tf,N)
#----------Matrices de Pauli---
pauli = pauli()
#-------Hamiltoniano-----------
h_cold  = 1
h_hot = 1.5
beta_hot = 1
beta_cold = 10
H_hot = h_hot/2.0 * pauli.z
H_cold = np.array([[h_cold/2.0, 0],[0, -h_cold/2.0]], dtype = np.complex)
H_constant = []

for i in range(0,N):
    H_constant.append(H_hot)
tau = 20*td
H_driven = []
H_driven.append(H_hot)
for i in range(0, 2*Nd, 2):
    H_driven.append((h_hot/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),pauli.z) + np.dot(np.exp(-t[i]/tau)*np.eye(2),pauli.x)))
    H_driven.append((h_hot/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),pauli.z) + np.dot(np.exp(-t[i]/tau)*np.eye(2),pauli.x)))
for i in range(2*Nd, N):
    H_driven.append(H_hot)
eps = np.sqrt(0.5)
V = eps*(np.kron(pauli.pl,pauli.pl) + np.kron(pauli.mn,pauli.mn))

bath = Reservoir(H_hot, beta_hot) 
qubit = System(H_driven, 2)
qubit.state.append(bath.Thermal_state())
qubit.energy.append(np.trace(np.dot(qubit.state[0], qubit.hamiltonian[0])))
qubit.work.append(0)
qubit.heat.append(0)
qubit.Driven_evolution(bath.Thermal_state(), bath.hamiltonian, V, td, dt)
qubit.Open_evolution(bath.Thermal_state(), bath.hamiltonian, V, tf-2*td, dt)
qubit.energy[0] = 0
qubit.work[1] = 0

plt.figure()
plt.plot(t, qubit.energy[:-1])
plt.plot(t, qubit.heat[:-1])
plt.plot(t, qubit.work[:-1])
plt.show()