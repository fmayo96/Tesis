import numpy as np
import matplotlib.pyplot as plt 
from repeatedinteractions_c import Reservoir, System

#------------Time--------------
tf = 10
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
h  = 1.5
H = np.array([[h/2.0, 0],[0, -h/2.0]], dtype = np.complex)
H_two_qubits = np.array(np.kron(H, Id) + np.kron(Id, H))
H_constant = []
for i in range(N):
    H_constant.append(H)
H_constant2 = []
for i in range(N):
    H_constant2.append(np.kron(H, Id) + np.kron(Id, H))
tau = 20*td
H_driven = [H]

for i in range(0,2*Nd,2):
    H_driven.append((h/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),sz) + np.dot(np.exp(-t[i]/tau)*np.eye(2),sx)))
    H_driven.append((h/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),sz) + np.dot(np.exp(-t[i]/tau)*np.eye(2),sx)))
for i in range(2*Nd, N):
    H_driven.append(H)
H_driven2 = []
for i in range(N):
    H_driven2.append(np.kron(H_driven[i], Id) + np.kron(Id, H_driven[i]))
eps = np.sqrt(0.5)
V = np.array(eps*(np.kron(su,su) + np.kron(sd,sd)))
V2 = np.array(2* eps * (np.kron(su, np.kron(su, np.kron(su, su))) + np.kron(sd, np.kron(sd, np.kron(sd, sd))))) 
#print(V2)
V_db = np.array(eps*(np.kron(su,sd) + np.kron(sd,su)))
#--------Condiciones Iniciales------------
bath = Reservoir(H, 1)
two_qubits_bath = Reservoir(H_two_qubits, 1)
qubit = System(H_driven, 2)
two_qubits = System(H_driven2, 4)
qubit.state.append(bath.Thermal_state())
qubit.energy.append(np.trace(np.dot(qubit.state[0], qubit.hamiltonian[0])))
qubit.work.append(0)
qubit.heat.append(0)
two_qubits.state.append(two_qubits_bath.Thermal_state())
two_qubits.energy.append(np.trace(np.dot(two_qubits.state[0], two_qubits.hamiltonian[0])))
two_qubits.work.append(0)
two_qubits.heat.append(0)
#---------Evolución-----------------------
qubit.Driven_evolution(bath.Thermal_state(), bath.hamiltonian, V, td, dt)
qubit.Open_evolution(bath.Thermal_state(), bath.hamiltonian, V, tf-2*td, dt)
two_qubits.Driven_evolution(two_qubits_bath.Thermal_state(), two_qubits_bath.hamiltonian, V2, td, dt)
two_qubits.Open_evolution(two_qubits_bath.Thermal_state(), two_qubits_bath.hamiltonian, V2, tf-2*td, dt)
qubit.work[1] = 0
qubit.energy[0] = 0
two_qubits.work[1] = 0
two_qubits.energy[0] = 0
print(qubit.state[-1])
print(two_qubits.state[-1])
with open('two_qubits_state.txt', 'w') as f:
    for i in range(N):
        f.write("{} \n".format(two_qubits.state[i]))
with open('qubit_state.txt', 'w') as f:
    for i in range(N):
        f.write("{} \n".format(qubit.state[i]))

#--------Graficar---------------------------
t = np.linspace(0, qubit.stroke_count * N, len(qubit.energy))
plt.figure()
plt.plot(t, qubit.energy, 'C0-.', linewidth = 2)
plt.plot(t, qubit.heat,'r-.', linewidth = 2)
plt.plot(t, qubit.work, 'C7-.', linewidth = 2)
plt.plot(t, np.array(two_qubits.energy)/2.0, 'C0', linewidth = 2)
plt.plot(t, np.array(two_qubits.heat)/2.0,'r', linewidth = 2)
plt.plot(t, np.array(two_qubits.work)/2.0, 'C7', linewidth = 2)
plt.legend(["E", "Q", "W","E_corr", "Q_corr", "W_corr"], fontsize = 11)
plt.xlabel("Time", fontsize = 12)
plt.ylabel("Energy", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

plt.figure()
plt.plot(t, np.array(qubit.state)[:,0,0])
plt.plot(t, np.array(qubit.state)[:,1,1])
plt.plot(t, np.array(two_qubits.state)[:,[0,0,0,0]])
plt.plot(t, np.array(two_qubits.state)[:,[3,3,3,3]])

plt.show()