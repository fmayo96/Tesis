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
h_hot = [6]
beta_hot = [0.01]
beta_cold = 10
eff_constant_temps = []
eff_driven_temps = []
for j in range(len(beta_hot)):
    driven_work = []
    constant_work = []

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
        V_db = eps*(np.kron(pauli.pl,pauli.mn) + np.kron(pauli.mn,pauli.pl))
        No_int = np.zeros(np.shape(V))
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
        driven_work.append(qubit_dr.work[-1])
        constant_work.append(qubit.work[-1])
    W_W3 = np.zeros(len(h_hot))
    eff_driven = np.zeros(len(h_hot))
    eff_constant = np.zeros(len(h_hot))
    
    Q_hot = np.zeros(len(h_hot))
    for i in range(len(h_hot)):
        Q_hot[i] = h_hot[i]*(np.tanh(beta_cold*h_cold/2.0) - np.tanh(beta_hot[j]*h_hot[i]/2.0))/2.0
        W_W3[i] = (h_hot[i] + h_cold)*np.tanh(beta_cold*h_cold/2.0)/2.0 - (h_hot[i] - h_cold)*np.tanh(beta_hot[j]*h_hot[i]/2.0)/2.0  
    for i in range(len(h_hot)):    
        eff_driven[i] = (W_W3[i] + driven_work[i])/Q_hot[i]
        eff_constant[i] = (W_W3[i] + constant_work[i])/Q_hot[i]
        eff_driven_temps.append(eff_driven[i])
        eff_constant_temps.append(eff_constant[i])
        

eff_driven_beta1 = eff_driven_temps[0:len(h_hot)]
eff_driven_beta2 = eff_driven_temps[len(h_hot):2*len(h_hot)]
eff_driven_beta3 = eff_driven_temps[2*len(h_hot):3*len(h_hot)]
eff_driven_beta4 = eff_driven_temps[3*len(h_hot):4*len(h_hot)]
eff_constant_beta1 = eff_constant_temps[0:len(h_hot)]
eff_constant_beta2 = eff_constant_temps[len(h_hot):2*len(h_hot)]
eff_constant_beta3 = eff_constant_temps[2*len(h_hot):3*len(h_hot)]
eff_constant_beta4 = eff_constant_temps[3*len(h_hot):4*len(h_hot)]


plt.figure()
plt.plot(h_hot, eff_driven_beta1, 'C0', linewidth = 2)
plt.plot(h_hot, eff_driven_beta2,'C1', linewidth = 2)
plt.plot(h_hot, eff_driven_beta3,'C2', linewidth = 2)
plt.plot(h_hot, eff_driven_beta4,'C3', linewidth = 2)
plt.plot(h_hot, 1-(h_cold/h_hot),'C7--', linewidth=2)
plt.plot(h_hot, eff_constant_beta1, 'C0--',linewidth = 2)
plt.plot(h_hot, eff_constant_beta2,'C1--', linewidth = 2)
plt.plot(h_hot, eff_constant_beta3,'C2--', linewidth = 2)
plt.plot(h_hot, eff_constant_beta4, 'C3--',linewidth = 2)
plt.legend(["beta1", "beta2", "beta3", "beta4", 'Otto'], fontsize = 11)
plt.xlabel("h_hot")
plt.ylabel("Efficiency")
plt.show()



"""
#-------------Graficar--------------------
t = np.linspace(0, qubit.stroke_count * tf, len(qubit.energy))
plt.figure()
plt.plot(t, qubit.energy, 'C0-.', linewidth = 2)
plt.plot(t, qubit.heat,'r-.', linewidth = 2)
plt.plot(t, qubit.work, 'C7-.', linewidth = 2)
plt.plot(t, qubit_dr.energy, 'C0', linewidth = 2)
plt.plot(t, qubit_dr.heat,'r', linewidth = 2)
plt.plot(t, qubit_dr.work, 'C7', linewidth = 2)
plt.legend(["E", "Q", "W","E_driven", "Q_driven", "W_driven"], fontsize = 11)
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
"""