import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from RK4 import RK4
#------------Time--------------
tf = 10
dt = 0.001
N = int(tf/dt)
prev_steps = 0
stroke_count = 0
#----------Matrices de Pauli---
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
sx = [[0,1],[1,0]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]
#-------Hamiltoniano-----------
h = 1.5
H = np.array([[h/2.0, 0],[0, -h/2.0]])
eps=  np.sqrt(0.5)
V = eps*(np.kron(su,su) + np.kron(sd,sd))
V_db = eps*(np.kron(su,sd) + np.kron(sd,su))
#-------Sistema y Reservorios--
class Reservoir():
    def __init__(self, hamiltonian, beta):
        self.hamiltonian = hamiltonian
        self.beta = beta
    def Thermal_state(self):        
        return np.diag(np.diag(np.exp(-self.beta * self.hamiltonian))) / np.sum(np.diag(np.diag(np.exp(-self.beta * self.hamiltonian))))
    def Active_state(self):        
        return np.diag(np.diag(np.exp(self.beta * self.hamiltonian))) / np.sum(np.diag(np.diag(np.exp(self.beta * self.hamiltonian))))
bath = Reservoir(H, 1)

class System():
    def __init__(self, hamiltonian):
        self.state = []
        self.hamiltonian = hamiltonian
        self.energy =[] 
        self.work = []
        self.heat = []
    def Dissipator(self, bath, V):
            rho = np.kron(self.state[-1], bath)
            Conmutator = np.dot(V,np.dot(V,rho))-np.dot(V,np.dot(rho,V))-np.dot(V,np.dot(rho,V))+np.dot(rho,np.dot(V,V))
            return -0.5*np.trace(Conmutator.reshape([2,2,2,2]), axis1 = 1, axis2 = 3)
    def Propagator(self, state, bath, V):
            return -1j * (np.dot(self.hamiltonian, state) - np.dot(state, self.hamiltonian)) + self.Dissipator(bath, V)
    def Time_evolution(self, bath, bath_hamiltonian, tf, dt, V, prev_steps, stroke_count):
        stroke_count += 1
        N = int(tf/dt)
        for i in tqdm(range(prev_steps, prev_steps +  N - 1)):
            self.state.append(RK4(self.Propagator, self.state[-1], bath, dt, V))
        for i in tqdm(range(prev_steps, prev_steps + N - 1)):
            self.energy.append(np.trace(np.dot(self.state[i], self.hamiltonian)) - self.energy[0])
            rhop = np.kron(self.state[i],bath)
            Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),bath_hamiltonian)))-np.dot(V,np.dot(np.kron(np.eye(2),bath_hamiltonian),V))-np.dot(V,np.dot(np.kron(np.eye(2),bath_hamiltonian),V))+np.dot(np.kron(np.eye(2),bath_hamiltonian),np.dot(V,V))
            self.heat.append(self.heat[-1] + dt * np.trace(np.dot(rhop, Conmutator)) * 0.5)
            self.work.append(self.energy[i] - self.heat[i])
        return prev_steps + N
#--------Condiciones Iniciales------------
qubit = System(H)
qubit.state.append(bath.Thermal_state())
qubit.energy.append(np.trace(np.dot(qubit.state[0], qubit.hamiltonian)))
qubit.heat.append(0)
qubit.work.append(0)
#---------Evolución-----------------------
prev_steps = qubit.Time_evolution(bath.Thermal_state(), bath.hamiltonian, tf, dt, V, prev_steps, stroke_count)
prev_steps = qubit.Time_evolution(bath.Thermal_state(), bath.hamiltonian, tf, dt, V_db, prev_steps, stroke_count)
print(f"Stroke_count = {stroke_count}")
qubit.energy[0] = 0
qubit.work[1] = 0
t = np.linspace(0, 2 * tf, 2 * N - 1)
#-------------Graficar--------------------
plt.figure()
plt.plot(t, qubit.energy, linewidth=2)
plt.plot(t, qubit.heat, linewidth=2)
plt.plot(t, qubit.work, 'C7', linewidth=2)
plt.legend(["E", "Q", "W"])
plt.show()



"""
rho_00 = []
rho_01 = []
rho_10 = []
rho_11 = []

for i in range(N):
    rho_00.append(qubit.state[i][0,0])
    rho_01.append(qubit.state[i][0,1])
    rho_10.append(qubit.state[i][1,0])
    rho_11.append(qubit.state[i][1,1])

plt.figure()
plt.plot(t, rho_00, linewidth=2)
plt.plot(t, rho_01, linewidth=2)
plt.plot(t, rho_10, linewidth=2)
plt.plot(t, rho_11, linewidth=2)
plt.show()
"""