import numpy as np
import matplotlib.pyplot as plt 
from repeatedintereactions import Reservoir, System
#------------Time--------------
tf = 15
dt = 0.001
N = int(tf/dt)
td = 0.85
Nd = int(td / dt)
t = np.linspace(0,tf,N)
#----------Matrices de Pauli---
Id = np.array([[1,0],[0,1]], dtype = np.complex)
sz = np.array([[1,0],[0,-1]], dtype = np.complex)
sx = np.array([[0,1],[1,0]], dtype = np.complex)
su = np.array([[0,0],[1,0]], dtype = np.complex)
sd = np.array([[0,1],[0,0]], dtype = np.complex)
#-------Hamiltoniano-----------
h_hot  = 1.5
h_cold = 1
H_hot = np.array([[h_hot/2.0, 0],[0, -h_hot/2.0]], dtype = np.complex)
H_cold = np.array([[h_cold/2.0, 0],[0, -h_cold/2.0]], dtype = np.complex)
H_constant = []
for i in range(N):
    H_constant.append(H_hot)
for i in range(N, 2*N):
    H_constant.append(H_cold)
tau = 20*td
H_driven = []
for i in range(0, 2*Nd, 2):
    H_driven.append((h_hot/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),sz) + np.dot(np.exp(-t[i]/tau)*np.eye(2),sx)))
    H_driven.append((h_hot/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),sz) + np.dot(np.exp(-t[i]/tau)*np.eye(2),sx)))
for i in range(2*Nd, N):
    H_driven.append(H_hot)
for i in range(N, 2*N):
    H_driven.append(H_cold)
H_driven[-2] = H_hot
H_constant[-2] = H_hot
eps = np.sqrt(0.5)
V = eps*(np.kron(su,su) + np.kron(sd,sd))
V_db = eps*(np.kron(su,sd) + np.kron(sd,su))
#--------Condiciones Iniciales------------
cold_bath = Reservoir(H_cold, 10)
hot_bath = Reservoir(H_hot,1)
qubit = System(H_constant, 2)
qubit.state.append(hot_bath.Thermal_state())
qubit.energy.append(np.trace(np.dot(qubit.state[0], qubit.hamiltonian[0])))
qubit.heat.append(0)
qubit.work.append(0)
qubit_dr = System(H_driven, 2)
qubit_dr.state.append(hot_bath.Thermal_state())
qubit_dr.energy.append(np.trace(np.dot(qubit.state[0], H_hot)))
qubit_dr.heat.append(0)
qubit_dr.work.append(0)
#---------Evolución-----------------------
qubit_dr.Driven_evolution(hot_bath.Thermal_state(), hot_bath.hamiltonian, td, dt, V)
qubit_dr.Time_evolution(hot_bath.Thermal_state(), hot_bath.hamiltonian, tf - 2 * td, dt, V)
qubit.Time_evolution(hot_bath.Thermal_state(), hot_bath.hamiltonian, tf, dt, V)
qubit.energy[0] = 0
qubit_dr.energy[0] = 0
#-------------Graficar--------------------
t = np.linspace(0, qubit_dr.stroke_count * N, len(qubit_dr.energy))
print(f"work len = {len(qubit_dr.work)}")
print(f"energy len = {len(qubit_dr.energy)}")
plt.figure()
plt.plot(t, qubit.energy, 'C0-.', linewidth = 2)
plt.plot(t, qubit.heat,'r-.', linewidth = 2)
plt.plot(t, qubit.work, 'C7-.', linewidth = 2)
plt.plot(t, qubit_dr.energy, 'C0', linewidth = 2)
plt.plot(t, qubit_dr.heat,'r', linewidth = 2)
plt.plot(t, qubit_dr.work, 'C7', linewidth = 2)
plt.legend(["E", "Q", "W","E_dr", "Q_dr", "W_dr"], fontsize = 11)
plt.xlabel("Time", fontsize = 12)
plt.ylabel("Energy", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()




rho_00 = []
rho_01 = []
rho_10 = []
rho_11 = []

for i in range(len(t)):
    rho_00.append(qubit_dr.state[i][0,0])
    rho_01.append(qubit_dr.state[i][0,1])
    rho_10.append(qubit_dr.state[i][1,0])
    rho_11.append(qubit_dr.state[i][1,1])

plt.figure()
plt.plot(t, rho_00, linewidth=2)
plt.plot(t, rho_01, linewidth=2)
plt.plot(t, rho_10, linewidth=2)
plt.plot(t, rho_11, linewidth=2)
plt.legend(["rho_00","rho_01","rho_10","rho_11"])
plt.show()
