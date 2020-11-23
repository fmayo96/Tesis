import numpy as np 
import matplotlib.pyplot as plt 
from repeatedinteractions_c import *
#------------Time--------------
tf = np.linspace(3,18,30)
WvsTf_dr = []
WvsTf = []
Q_h_dr = []
Q_h = []
pot = []
pot_dr = []
for j in range(len(tf)):
    dt = 0.001
    N = int(tf[j]/dt)
    td = 0.89
    Nd = int(td / dt)
    t = np.linspace(0,tf[j],N)
    #----------Matrices de Pauli---
    Id = np.array([[1,0],[0,1]], dtype = np.complex)
    sz = np.array([[1,0],[0,-1]], dtype = np.complex)
    sx = np.array([[0,1],[1,0]], dtype = np.complex)
    su = np.array([[0,0],[1,0]], dtype = np.complex)
    sd = np.array([[0,1],[0,0]], dtype = np.complex)
    #-------Hamiltoniano-----------
    h_hot  = 6
    h_cold = 1
    H_hot = np.array([[h_hot/2.0, 0],[0, -h_hot/2.0]], dtype = np.complex)
    H_cold = np.array([[h_cold/2.0, 0],[0, -h_cold/2.0]], dtype = np.complex)
    H_constant = []
    for i in range(N):
        H_constant.append(H_cold)
    tau = 20*td
    H_driven = []
    for i in range(0, 2*Nd, 2):
        H_driven.append((h_cold/2)*(sx))
        H_driven.append((h_cold/2)*(sx))
    if (tf[j] > 2*td):
        for i in range(2*Nd, N):
            H_driven.append(H_cold)
        for i in range(N, 2*N):
            H_driven.append(H_cold)
    eps = np.sqrt(0.5)
    V = eps*(np.kron(su,su) + np.kron(sd,sd))
    V_db = eps*(np.kron(su,sd) + np.kron(sd,su))
    #--------Condiciones Iniciales------------
    cold_bath = Reservoir(H_cold, 10)
    hot_bath = Reservoir(H_hot,0.2)
    qubit = System(H_constant, 2)
    qubit.state.append(hot_bath.Thermal_state())
    qubit.energy.append(np.trace(np.dot(qubit.state[0], qubit.hamiltonian[0])))
    qubit.heat.append(0)
    qubit.work.append(0)
    qubit_dr = System(H_driven, 2)
    qubit_dr.state.append(hot_bath.Thermal_state())
    qubit_dr.energy.append(np.trace(np.dot(qubit.state[0], H_cold)))
    qubit_dr.heat.append(0)
    qubit_dr.work.append(0)
    #---------Evolución-----------------------
    if (tf[j] > 2*td):
        qubit_dr.Driven_evolution(cold_bath.Thermal_state(), cold_bath.hamiltonian, V, td, dt)
        qubit_dr.Open_evolution(cold_bath.Thermal_state(), cold_bath.hamiltonian, V, tf[j] - 2 * td, dt)
    else:
        qubit_dr.Driven_evolution(cold_bath.Thermal_state(), cold_bath.hamiltonian, V, tf[j]/2.0, dt)
    qubit.Open_evolution(cold_bath.Thermal_state(), cold_bath.hamiltonian, V, tf[j], dt)
    qubit.energy[0] = 0
    qubit_dr.energy[0] = 0
#----------Trabajos-----------------------
    rho_1 = hot_bath.Active_state()
    rho_2 = hot_bath.Thermal_state()
    W_12 = -np.trace(np.dot(H_hot,rho_2 - rho_1))
    W_23 = -np.trace(np.dot(H_cold - H_hot,rho_2))
    W_45_dr = -np.trace(np.dot(H_hot - H_cold,qubit_dr.state[-1]))
    W_45 = -np.trace(np.dot(H_hot - H_cold,qubit.state[-1]))
    W_56_dr = -2*np.trace(np.dot(H_hot,rho_1 - qubit_dr.state[-1]))
    W_56 = -2*np.trace(np.dot(H_hot,rho_1 - qubit.state[-1]))
    WvsTf_dr.append(qubit_dr.work[-1] + W_12 + W_23 + W_45_dr + W_56_dr)
    WvsTf.append(qubit.work[-1] + W_12 + W_23 + W_45 + W_56)
    pot.append((WvsTf[-1])/tf[j])
    pot_dr.append((WvsTf_dr[-1])/tf[j])
    Q_h.append(0.5*W_56)
    Q_h_dr.append(0.5*W_56_dr)
#-------------Graficar--------------------

t = np.linspace(0, tf[-1], len(qubit_dr.energy))

print(f"work len = {len(qubit_dr.work)}")
print(f"energy len = {len(qubit_dr.energy)}")
"""
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
"""

eff_dr = []
eff = []
otto_eff = (1 - h_cold/h_hot) * np.ones(len(tf))
for i in range(len(tf)):
   eff_dr.append(WvsTf_dr[i])
   eff.append(WvsTf[i]) 
plt.figure()
plt.plot(tf, eff_dr, linewidth=2)
plt.plot(tf, eff, linewidth=2)
plt.plot(tf, otto_eff, 'C7--', linewidth=2)
plt.plot(tf, np.zeros(len(tf)), 'r--')
plt.title("beta_hot = 0.2 h_hot = 6", fontsize=12)
plt.legend(["Eff driven", "Eff constant", "Eff Otto"], fontsize=11)
plt.xlabel('Time', fontsize=12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize=11)
plt.show()

plt.figure()
plt.plot(tf, pot_dr, linewidth=2)
plt.plot(tf, pot, linewidth=2)
plt.plot(tf, np.zeros(len(tf)), 'r--')
plt.title("beta_hot = 0.2 h_hot = 6", fontsize=12)
plt.legend(["Pot driven", "Pot constant"], fontsize=11)
plt.xlabel('Time', fontsize=12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize=11)
plt.show()





plt.figure()
plt.plot(pot_dr, eff_dr/otto_eff, linewidth=2)
plt.plot(pot, eff/otto_eff, linewidth=2)
plt.title("beta_hot = 0.2 h_hot = 6", fontsize=12)
plt.legend(["Driven", "Constant"], fontsize=11)
plt.xlabel('Power', fontsize=12)
plt.ylabel("Eff/Eff otto")
plt.xticks(fontsize = 11)
plt.yticks(np.linspace(0,1,6),fontsize=11)
plt.show()
"""
eff_dr_file = open("eff_dr_13-18.txt", "w")
np.savetxt(eff_dr_file, eff_dr)
eff_dr_file.close()
eff_file = open("eff_13-18.txt", "w")
np.savetxt(eff_file, eff)
eff_file.close()
pot_dr_file = open("pot_dr_13-18.txt", "w")
np.savetxt(pot_dr_file, pot_dr)
pot_dr_file.close()
pot_file = open("pot_13-18.txt", "w")
np.savetxt(pot_file, pot)
pot_file.close()
"""