import numpy as np
import matplotlib.pyplot as plt 
from repeatedinteractions_c import Reservoir, System
from pauli import pauli
#------------Time--------------
tf = 20
dt = 0.001
N = int(tf/dt)
td = 0.85
Nd = int(td / dt)
t = np.linspace(0,tf,N)
#----------Matrices de Pauli---
pauli = pauli()
#-------Hamiltoniano-----------
h_cold  = 1
h_hot = [6]
beta_hot = [0.01]
beta_cold = 10
pow_constant_temps = []
pow_driven_temps = []
i_final = 0
for j in range(len(beta_hot)):
    driven_power = []
    constant_power = []

    for i in range(len(h_hot)):
            
        H_hot = np.array([[h_hot[i]/2.0, 0],[0, -h_hot[i]/2.0]], dtype = np.complex)
        H_cold = np.array([[h_cold/2.0, 0],[0, -h_cold/2.0]], dtype = np.complex)
        H_constant = []

        for i in range(0,N):
            H_constant.append(H_cold)
        tau = 20*td
        H_driven = []
        H_driven.append(H_cold)
        for i in range(0, 2*Nd, 2):
            H_driven.append((h_cold/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),pauli.z) + np.dot(np.exp(-t[i]/tau)*np.eye(2),pauli.x)))
            H_driven.append((h_cold/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),pauli.z) + np.dot(np.exp(-t[i]/tau)*np.eye(2),pauli.x)))
        for i in range(2*Nd, N):
            H_driven.append(H_cold)
        eps = np.sqrt(0.5)
        V = eps*(np.kron(pauli.pl,pauli.pl) + np.kron(pauli.mn,pauli.mn))
        #--------Condiciones Iniciales------------
        cold_bath = Reservoir(H_cold, beta_cold)
        hot_bath = Reservoir(H_hot,beta_hot[j])
        qubit_dr = System(H_driven, 2)
        qubit_dr.state.append(hot_bath.Thermal_state())
        qubit_dr.energy.append(np.trace(np.dot(qubit_dr.state[0], qubit_dr.hamiltonian[0])))
        qubit_dr.heat.append(0)
        qubit_dr.work.append(0)
        qubit = System(H_constant, 2)
        qubit.state.append(hot_bath.Thermal_state())
        qubit.energy.append(np.trace(np.dot(qubit_dr.state[0], qubit_dr.hamiltonian[0])))
        qubit.work.append(0)
        qubit.heat.append(0)
        #---------Evolución-----------------------
        qubit.Open_evolution(cold_bath.Thermal_state(), cold_bath.hamiltonian, V, tf, dt)
        qubit_dr.Driven_evolution(cold_bath.Thermal_state(), cold_bath.hamiltonian, V, td, dt)
        qubit_dr.Open_evolution(cold_bath.Thermal_state(), cold_bath.hamiltonian, V, tf-2*td, dt)
        qubit_dr.energy[0] = 0
        qubit.energy[0] = 0
        qubit_dr.work[1] = 0
        qubit.work[1] = 0
        for i in range(len(qubit_dr.work)):
            if qubit_dr.work[i] >= 0.95 * qubit_dr.work[-1]:
                i_final = i
                
            else:
                driven_power.append(qubit_dr.work[i_final]/t[i_final])
                
                break
        
        for i in range(len(qubit.work)):
            if qubit.work[i] >= 0.95 * qubit.work[-1]:
                i_final = i
                
            else:
                constant_power.append(qubit.work[i_final]/t[i_final])
                
                break
    for i in range(len(h_hot)):
        pow_driven_temps.append(driven_power[i])
        pow_constant_temps.append(constant_power[i])
        print(pow_constant_temps)
pow_driven_beta1 = pow_driven_temps[0:len(h_hot)]
pow_driven_beta2 = pow_driven_temps[len(h_hot):2*len(h_hot)]
pow_driven_beta3 = pow_driven_temps[2*len(h_hot):3*len(h_hot)]
pow_driven_beta4 = pow_driven_temps[3*len(h_hot):4*len(h_hot)]
pow_constant_beta1 = pow_constant_temps[0:len(h_hot)]
pow_constant_beta2 = pow_constant_temps[len(h_hot):2*len(h_hot)]
pow_constant_beta3 = pow_constant_temps[2*len(h_hot):3*len(h_hot)]
pow_constant_beta4 = pow_constant_temps[3*len(h_hot):4*len(h_hot)]

delta_p1 = []
delta_p2 = []
delta_p3 = []
delta_p4 = []

for i in range(len(h_hot)):
    delta_p1.append((-pow_constant_beta1[i] + pow_driven_beta1[i])/abs(pow_driven_beta1[i]))
    delta_p2.append((-pow_constant_beta2[i] + pow_driven_beta2[i])/abs(pow_driven_beta2[i]))
    delta_p3.append((-pow_constant_beta3[i] + pow_driven_beta3[i])/abs(pow_driven_beta3[i]))
    delta_p4.append((-pow_constant_beta4[i] + pow_driven_beta4[i])/abs(pow_driven_beta4[i]))

plt.figure()
plt.plot(h_hot, delta_p1, 'C0', linewidth = 2)
plt.plot(h_hot, delta_p2,'C1', linewidth = 2)
plt.plot(h_hot, delta_p3,'C2', linewidth = 2)
plt.plot(h_hot, delta_p4,'C3', linewidth = 2)
plt.legend(["beta1", "beta2", "beta3", "beta4"], fontsize = 11)
plt.xlabel("h_hot")
plt.ylabel("Power")
plt.show()
