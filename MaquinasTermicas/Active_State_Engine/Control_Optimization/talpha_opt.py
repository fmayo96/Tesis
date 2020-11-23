import numpy as np
import matplotlib.pyplot as plt
from repeatedinteractions_c import Reservoir, System
from pauli import *
pauli = pauli()
tf = 10
dt = 0.001
t_alpha = np.linspace(0.8, 1.5, 100)




N = int(tf/dt)
E_alpha = []
for j in range(len(t_alpha)):
    N_alpha = int(t_alpha[j]/dt)
    h = 1.5
    beta = 1
    H_bath = h/2*pauli.z
    H = np.zeros([N,2,2])
    H[0] = h/2*pauli.z
    for i in range(1,N_alpha*2):
        H[i] = h/2*pauli.x
    for i in range(2*N_alpha, N):
        H[i] = h/2*pauli.z
    eps = np.sqrt(0.5)
    V = eps*(np.kron(pauli.pl,pauli.pl) + np.kron(pauli.mn,pauli.mn))
    bath = Reservoir(H_bath, beta)
    qubit = System(H, 2)
    qubit.state.append(bath.Thermal_state())
    qubit.energy.append(np.trace(np.dot(qubit.state[0], qubit.hamiltonian[0])))
    qubit.work.append(0)
    qubit.heat.append(0)
    qubit.Driven_evolution(bath.Thermal_state(), bath.hamiltonian, V, t_alpha[j], dt)
    qubit.Open_evolution(bath.Thermal_state(), bath.hamiltonian, V, tf- 2*t_alpha[j], dt)
    qubit.energy[0] = 0
    E_alpha.append(qubit.energy[2*N_alpha + 1])


print(np.argmax(E_alpha))

plt.figure()
plt.plot(t_alpha, E_alpha, linewidth=2)
plt.xlabel("t_alpha", fontsize=12)
plt.ylabel("Energy (t_alpha)", fontsize=12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.show()