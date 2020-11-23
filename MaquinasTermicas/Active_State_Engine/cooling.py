import numpy as np 
import matplotlib.pyplot as plt 
from repeatedinteractions_c import Reservoir, System
from pauli import pauli
#------------Time--------------
tf = 10
dt = 0.001
N = int(tf/dt)
t = np.linspace(0,tf,N)
#----------Matrices de Pauli---
pauli = pauli()
#--------Hamiltonianos--------
h = 1
h_bath = 5
beta = 0.25
W_quench_const = h/2.0 * (5) * np.tanh(beta*h/2.0)
W_quench = np.ones(N)*W_quench_const
eps = np.sqrt(5)
H_bath = h_bath/2.0*pauli.z 
H_aux = h/2.0 * pauli.z
H = np.zeros([N,2,2], dtype = np.complex)
for i in range(N):
    H[i] = h/2.0*pauli.z 
V = eps*(np.kron(pauli.pl,pauli.mn) + np.kron(pauli.mn,pauli.pl))
#--------Condiciones iniciales-------
bath = Reservoir(H_bath, beta)
aux_bath = Reservoir(H_aux, beta)
qubit = System(H, 2)
qubit.state.append(aux_bath.Thermal_state())
qubit.energy.append(np.trace(np.dot(qubit.state[0],qubit.hamiltonian[0])))
qubit.heat.append(0)
qubit.work.append(0)
#-------Evolucion temporal------------
qubit.Open_evolution(bath.Thermal_state(), bath.hamiltonian, V, tf, dt)
qubit.energy[0] = 0
qubit.work[0] = 0
qubit.work[1] = 0
for i in range(N):
    qubit.work[i] = -1*qubit.work[i] 
print(f"COP = {np.abs(qubit.heat[-1])/qubit.work[-1]}")
#-------Graficar----------------------

plt.figure()
plt.plot(t,qubit.energy[:], linewidth=2)
plt.plot(t,qubit.heat[:], linewidth=2)
plt.plot(t,qubit.work[:], linewidth=2)
plt.plot(t, W_quench, linewidth=2)
plt.legend(["Energy", "Heat", "Work"], fontsize=11)
plt.xlabel("Time", fontsize = 12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)

plt.show()

rho_00 = np.zeros(N)
rho_11 = np.zeros(N)
for i in range(N):
    rho_00[i] = qubit.state[i][0][0]
    rho_11[i] = qubit.state[i][1][1]
plt.figure()
plt.plot(t, rho_00)
plt.plot(t, rho_11)
plt.legend(["Excited state","Ground state"], fontsize = 11)
plt.xlabel("Time", fontsize = 12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.show()