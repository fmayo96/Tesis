import numpy as np 
import matplotlib.pyplot as plt 
from RK4 import RK4
from repeatedintereactions import Reservoir, System

#------------Time--------------
tf = 5
dt = 0.001
N = int(tf/dt)
t = np.linspace(0,tf,N)
#----------Matrices de Pauli---
Id = np.array([[1,0],[0,1]], dtype = np.complex)
sz = np.array([[1,0],[0,-1]], dtype = np.complex)
sx = np.array([[0,1],[1,0]], dtype = np.complex)
su = np.array([[0,0],[1,0]], dtype = np.complex)
sd = np.array([[0,1],[0,0]], dtype = np.complex)
#-------Hamiltoniano-----------
h = 1.5
H_bath = h/2.0 * sz 
H_bath2 = h/2.0 * (np.kron(sz, Id) + np.kron(Id, sz))
H_bath3 = h/2.0 * (np.kron(sz, np.kron(Id, Id)) + np.kron(Id, np.kron(sz, Id)) + np.kron(Id, np.kron(Id, sz)))
H_bath4 = h/2.0 * (np.kron(sz, np.kron(Id, np.kron(Id, Id))) + np.kron(Id, np.kron(sz, np.kron(Id, Id))) + np.kron(Id, np.kron(Id, np.kron(sz, Id))) + np.kron(Id, np.kron(Id, np.kron(Id, sz))))
H = []
H2 = []
H3 = []
H4 = []
for i in range(N):
    H.append(H_bath)
    H2.append(H_bath2)
    H3.append(H_bath3)
    H4.append(H_bath4)
eps = np.sqrt(5)
V = eps * (np.kron(su,su) + np.kron(sd,sd))
V2 = 2* eps * (np.kron(su,np.kron(su,np.kron(su,su))) + np.kron(sd,np.kron(sd,np.kron(sd,sd)))+np.kron(su,np.kron(sd,np.kron(su,sd))) + np.kron(sd,np.kron(su,np.kron(sd,su))))
V3 = 3 * eps * (np.kron(su,np.kron(su,np.kron(su,np.kron(su,np.kron(su,su))))) + np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,sd))))) + np.kron(su,np.kron(sd,np.kron(su,np.kron(su,np.kron(sd,su))))) + np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,np.kron(su,sd)))))+ np.kron(su,np.kron(su,np.kron(sd,np.kron(su,np.kron(su,sd))))) + np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,su)))))+ np.kron(sd,np.kron(su,np.kron(su,np.kron(sd,np.kron(su,su))))) + np.kron(su,np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,sd))))))
V4 = 4 * eps * (np.kron(su,np.kron(su,np.kron(su,np.kron(su,np.kron(su,np.kron(su,np.kron(su,su))))))) + np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,sd))))))) + np.kron(su,np.kron(su,np.kron(su,np.kron(sd,np.kron(su,np.kron(su,np.kron(su,sd))))))) + np.kron(sd,np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,np.kron(sd,su)))))))+ np.kron(su,np.kron(su,np.kron(sd,np.kron(su,np.kron(su,np.kron(su,np.kron(sd,su))))))) + np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(su,sd))))))) + np.kron(su,np.kron(sd,np.kron(su,np.kron(su,np.kron(su,np.kron(sd,np.kron(su,su))))))) + np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,sd)))))))+ np.kron(su,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,sd))))))) + np.kron(sd,np.kron(su,np.kron(su,np.kron(su,np.kron(sd,np.kron(su,np.kron(su,su))))))) )
aval4, avec4 = np.linalg.eig(V4)
aval3,avec3 = np.linalg.eig(V3)
aval2,avec2 = np.linalg.eig(V2)
aval,avec = np.linalg.eig(V)
print("Norma 4: ", np.max(abs(aval4))/4)
print("Norma 3: ", np.max(abs(aval3))/3)
print("Norma 2: ", np.max(abs(aval2))/2)
print("Norma 1:", np.max(abs(aval)))
#----------Sistemas y reservorios-----
bath = Reservoir(H_bath, 1)
bath2 = Reservoir(H_bath2, 1)
bath3 = Reservoir(H_bath3, 1)
bath4 = Reservoir(H_bath4, 1)
qubit = System(H, 2)
qubit.state.append(bath.Thermal_state())
qubit.energy.append(np.trace(np.dot(qubit.state[0], qubit.hamiltonian[0])))
qubit.heat.append(0)
qubit.work.append(0)
two_qubits = System(H2, 4)
two_qubits.state.append(bath2.Thermal_state())
two_qubits.energy.append(np.trace(np.dot(two_qubits.state[0], two_qubits.hamiltonian[0])))
two_qubits.heat.append(0)
two_qubits.work.append(0)
three_qubits = System(H3, 8)
three_qubits.state.append(bath3.Thermal_state())
three_qubits.energy.append(np.trace(np.dot(three_qubits.state[0], three_qubits.hamiltonian[0])))
three_qubits.heat.append(0)
three_qubits.work.append(0)
four_qubits = System(H4, 16)
four_qubits.state.append(bath4.Thermal_state())
four_qubits.energy.append(np.trace(np.dot(four_qubits.state[0], four_qubits.hamiltonian[0])))
four_qubits.heat.append(0)
four_qubits.work.append(0)
#---------------Evolución-------------
tengo_datos = False
if tengo_datos:
    with open("1qubit.txt", 'r') as file:
        for line in file:
            qubit.state.append(line)
    with open("2qubits.txt", 'r') as file:
        for line in file:
            two_qubits.state.append(line)
    with open("3qubits.txt", 'r') as file:
        for line in file:
            three_qubits.state.append(line)
    with open("4qubit.txt", 'r') as file:
        for line in file:
            four_qubits.state.append(line)
else:
    qubit.Time_evolution(bath.Thermal_state(), bath.hamiltonian, tf, dt, V)
    two_qubits.Time_evolution(bath2.Thermal_state(), bath2.hamiltonian, tf, dt, V2)
    three_qubits.Time_evolution(bath3.Thermal_state(), bath3.hamiltonian, tf, dt, V3)
    four_qubits.Time_evolution(bath4.Thermal_state(), bath4.hamiltonian, tf, dt, V4)
    qubit.energy[0] = 0
    qubit.work[1] = 0
    two_qubits.energy[0] = 0
    two_qubits.work[1] = 0
    three_qubits.energy[0] = 0
    three_qubits.work[1] = 0
    four_qubits.energy[0] = 0
    four_qubits.work[1] = 0
#----------Graficar-----------------
plt.figure()
plt.plot(t, qubit.energy, 'C0', linewidth=2)
plt.plot(t, two_qubits.energy, 'C1', linewidth=2)
plt.plot(t, three_qubits.energy, 'C4', linewidth=2)
plt.plot(t, four_qubits.energy, 'C7', linewidth=2)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Energy', fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(['1 qubit', '2 qubits', '3 qubits', '4 qubits'])
plt.show()
#--------------Guardar archivos--------
with open('1qubit.txt', 'w') as f:
    for i in range(len(qubit.energy)):
        f.write("{},{},{},{}\n".format(qubit.state[i],qubit.energy[i], qubit.heat[i], qubit.work[i]))
with open('2qubits.txt', 'w') as f:
    for i in range(len(two_qubits.energy)):
        f.write("{},{},{},{}\n".format(two_qubits.state[i],two_qubits.energy[i], two_qubits.heat[i], two_qubits.work[i]))
with open('3qubits.txt', 'w') as f:
    for i in range(len(three_qubits.energy)):
        f.write("{},{},{},{}\n".format(three_qubits.state[i],three_qubits.energy[i], three_qubits.heat[i], three_qubits.work[i]))
with open('4qubits.txt', 'w') as f:
    for i in range(len(four_qubits.state)):
        f.write("{},{},{},{}\n".format(four_qubits.state[i],four_qubits.energy[i], four_qubits.heat[i], four_qubits.work[i]))
