import numpy as np
import matplotlib.pyplot as plt 
from repeatedinteractions import Reservoir, System
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
h_cold  = 1
h_hot = 2
H_hot = np.array([[h_hot/2.0, 0],[0, -h_hot/2.0]], dtype = np.complex)
H_cold = np.array([[h_cold/2.0, 0],[0, -h_cold/2.0]], dtype = np.complex)
H_constant = []
h = []
for i in range(0,N):
    H_constant.append(H_cold)
    h.append(h_hot)
for i in range(N, 2*N):
    H_constant.append(H_hot)
    h.append(h_cold)


h[-1] = h_hot

tau = 20*td
H_driven = []
for i in range(0, N):
    H_driven.append(H_cold)
H_driven.append(H_hot)
for i in range(0, 2*Nd, 2):
    H_driven.append((h_hot/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),sz) + np.dot(np.exp(-t[i]/tau)*np.eye(2),sx)))
    H_driven.append((h_hot/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),sz) + np.dot(np.exp(-t[i]/tau)*np.eye(2),sx)))
for i in range(2*Nd, N):
    H_driven.append(H_hot)
H_driven[-2] = H_cold
H_constant[-2] = H_cold
eps = np.sqrt(0.5)
V = eps*(np.kron(su,su) + np.kron(sd,sd))
V_db = eps*(np.kron(su,sd) + np.kron(sd,su))
No_int = np.zeros(np.shape(V))
#--------Condiciones Iniciales------------
cold_bath = Reservoir(H_cold, 10)
hot_bath = Reservoir(H_hot,1)
qubit = System(H_driven, 2)
qubit.state.append(hot_bath.Active_state())
qubit.energy.append(np.trace(np.dot(qubit.state[0], qubit.hamiltonian[0])))
qubit.heat.append(0)
qubit.work.append(0)
#---------Evolución-----------------------
qubit.Time_evolution(cold_bath.Thermal_state(), cold_bath.hamiltonian, tf, dt, V_db)
qubit.Time_evolution(hot_bath.Thermal_state(), hot_bath.hamiltonian, tf, dt, V)
qubit.energy[0] = 0
#qubit_dr.energy[0] = 0
qubit.work[1] = 0
#qubit_dr.work[1] = 0
print(f"Eficiencia = {qubit.work[-1]/qubit.heat[N-1]}")
#-------------Graficar--------------------
t = np.linspace(0, qubit.stroke_count * N, len(qubit.energy))
plt.figure()
plt.plot(t, qubit.energy, 'C0-.', linewidth = 2)
plt.plot(t, np.zeros(len(t)))
plt.plot(t, qubit.heat,'r-.', linewidth = 2)
plt.plot(t, qubit.work, 'C7-.', linewidth = 2)
plt.legend(["E", "Q", "W"], fontsize = 11)
plt.xlabel("Time", fontsize = 12)
plt.ylabel("Energy", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()



plt.figure()
plt.plot(t, np.array(qubit.state)[:,0,0], linewidth=2)
plt.plot(t, np.array(qubit.state)[:,0,1], linewidth=2)
plt.plot(t, np.array(qubit.state)[:,1,0], linewidth=2)
plt.plot(t, np.array(qubit.state)[:,1,1], linewidth=2)
plt.legend(["rho_00","rho_01","rho_10","rho_11"])
plt.show()
