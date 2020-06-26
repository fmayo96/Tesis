import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from RK4 import RK4
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
h = 1.5
H = np.array([[h/2.0, 0],[0, -h/2.0]], dtype = np.complex)
H_constant = []
for i in range(N):
    H_constant.append(H)
tau = 20*td
H_driven = []

for i in range(0, 2*Nd, 2):
    H_driven.append((h/2)*((1-np.exp(-t[i]/tau))*sz + np.exp(-t[i]/tau)*sx))
    H_driven.append((h/2)*((1-np.exp(-t[i]/tau))*sz + np.exp(-t[i]/tau)*sx))
for i in range(2*Nd, N):
    H_driven.append(H)
eps = np.sqrt(0.5)
V = eps*(np.kron(su,su) + np.kron(sd,sd))
V_db = eps*(np.kron(su,sd) + np.kron(sd,su))
#-------Sistema y Reservorios--
class Reservoir():
    def __init__(self, hamiltonian, beta):
        self.hamiltonian = hamiltonian
        self.beta = beta
    def Thermal_state(self):        
        return np.array(np.diag(np.diag(np.exp(-self.beta * self.hamiltonian))) / np.sum(np.diag(np.diag(np.exp(-self.beta * self.hamiltonian)))), dtype = np.complex)
    def Active_state(self):        
        return np.array(np.diag(np.diag(np.exp(self.beta * self.hamiltonian))) / np.sum(np.diag(np.diag(np.exp(self.beta * self.hamiltonian)))), dtype = np.complex)
bath = Reservoir(H, 1)
class System():
    def __init__(self, hamiltonian):
        self.state = []
        self.hamiltonian = hamiltonian
        self.energy =[] 
        self.work = []
        self.heat = []
        self.prev_steps = 0
        self.stroke_count = 0
    def Dissipator(self, bath, V):
            rho = np.kron(self.state[-1], bath)
            Conmutator = np.dot(V,np.dot(V,rho))-np.dot(V,np.dot(rho,V))-np.dot(V,np.dot(rho,V))+np.dot(rho,np.dot(V,V))
            return -0.5*np.trace(Conmutator.reshape([2,2,2,2]), axis1 = 1, axis2 = 3)
    def Propagator(self, hamiltonian, state, bath, V):
            return -1j * (np.dot(hamiltonian, state) - np.dot(state, hamiltonian)) + self.Dissipator(bath, V)
    def Unitary(self, hamiltonian, state, bath, V, dtype = np.complex):
            return  -1j * (np.dot(hamiltonian, state) - np.dot(state, hamiltonian))
    def Time_evolution(self, bath, bath_hamiltonian, tf, dt, V):
        N = int(tf/dt)
        for i in tqdm(range(self.prev_steps, self.prev_steps +  N - 1)):
            self.state.append(RK4(self.Propagator, self.hamiltonian[i], self.state[-1], bath, dt, V))
        for i in tqdm(range(self.prev_steps, self.prev_steps + N - 1)):
            self.energy.append(np.trace(np.dot(self.state[i], self.hamiltonian[i])) - self.energy[0])
            rhop = np.kron(self.state[i],bath)
            Conmutator = 0.5 * (np.dot(V,np.dot(V,np.kron(np.eye(2),bath_hamiltonian)))-np.dot(V,np.dot(np.kron(np.eye(2),bath_hamiltonian),V))-np.dot(V,np.dot(np.kron(np.eye(2),bath_hamiltonian),V))+np.dot(np.kron(np.eye(2),bath_hamiltonian),np.dot(V,V)))
            self.heat.append(self.heat[-1] + dt * np.trace(np.dot(rhop, Conmutator)))
            self.work.append(self.energy[i] - self.heat[i])
        self.prev_steps += N
        self.stroke_count += 1
    def Driven_evolution(self, bath, bath_hamiltonian, td, dt, V):
        Nd = int(td/dt)
        repetidos = 0
        for i in tqdm(range(self.prev_steps, self.prev_steps +  2*Nd, 2)):
            self.state.append(RK4(self.Propagator, self.hamiltonian[i], self.state[-1], bath, dt, V))
            self.state.append(RK4(self.Unitary, self.hamiltonian[i], self.state[-1], bath, dt, V))
        for i in tqdm(range(self.prev_steps, self.prev_steps + 2 * Nd)):
            self.energy.append(np.trace(np.dot(self.state[i], self.hamiltonian[i])) - self.energy[0])
        for i in range(self.prev_steps, self.prev_steps + 2*Nd, 2):
            rhop = np.kron(self.state[i],bath)
            Conmutator = 0.5 * (np.dot(V,np.dot(V,np.kron(np.eye(2),bath_hamiltonian)))-np.dot(V,np.dot(np.kron(np.eye(2),bath_hamiltonian),V))-np.dot(V,np.dot(np.kron(np.eye(2),bath_hamiltonian),V))+np.dot(np.kron(np.eye(2),bath_hamiltonian),np.dot(V,V)))
            self.heat.append(self.heat[-1] + dt * np.trace(np.dot(rhop, Conmutator)))
            self.heat.append(self.heat[-1])
        for i in tqdm(range(self.prev_steps, self.prev_steps + 2*Nd)):
            self.work.append(self.energy[i] - self.heat[i])
        self.prev_steps += 2*Nd
#--------Condiciones Iniciales------------
qubit = System(H_constant)
qubit.state.append(bath.Thermal_state())
qubit.energy.append(np.trace(np.dot(qubit.state[0], qubit.hamiltonian[0])))
qubit.heat.append(0)
qubit.work.append(0)
qubit_dr = System(H_driven)
qubit_dr.state.append(bath.Thermal_state())
qubit_dr.energy.append(np.trace(np.dot(qubit.state[0], H)))
qubit_dr.heat.append(0)
qubit_dr.work.append(0)
#---------Evolución-----------------------
qubit_dr.Driven_evolution(bath.Thermal_state(), bath.hamiltonian, td, dt, V)
qubit_dr.Time_evolution(bath.Thermal_state(), bath.hamiltonian, tf - 2 * td, dt, V)
qubit.Time_evolution(bath.Thermal_state(), bath.hamiltonian, tf, dt, V)
#prev_steps = qubit.Time_evolution(bath.Thermal_state(), bath.hamiltonian, tf, dt, V_db, prev_steps, stroke_count)
print(f"Stroke_count = {qubit.stroke_count}")
qubit.energy[0] = 0
qubit.work[1] = 0
qubit_dr.energy[0] = 0
qubit_dr.work[1] = 0
#-------------Graficar--------------------
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

for i in range(N):
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
