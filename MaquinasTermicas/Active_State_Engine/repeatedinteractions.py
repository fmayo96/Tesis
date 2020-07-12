import numpy as np 
from tqdm import tqdm
from RK4 import RK4

class Reservoir():
    def __init__(self, hamiltonian, beta):
        self.hamiltonian = hamiltonian
        self.beta = beta
    def Thermal_state(self):        
        return np.array(np.diag(np.diag(np.exp(-self.beta * self.hamiltonian))) / np.sum(np.diag(np.diag(np.exp(-self.beta * self.hamiltonian)))), dtype = np.complex)
    def Active_state(self):        
        return np.array(np.diag(np.diag(np.exp(self.beta * self.hamiltonian))) / np.sum(np.diag(np.diag(np.exp(self.beta * self.hamiltonian)))), dtype = np.complex)
class System():
    def __init__(self, hamiltonian, dim):
        self.dim = dim
        self.state = []
        self.hamiltonian = hamiltonian
        self.energy =[] 
        self.work = []
        self.heat = []
        self.prev_steps = 0
        self.stroke_count = 0
    def Dissipator(self, bath, V):
            shape = self.dim
            rho = np.kron(self.state[-1], bath)
            Conmutator = np.dot(V,np.dot(V,rho))-np.dot(V,np.dot(rho,V))-np.dot(V,np.dot(rho,V))+np.dot(rho,np.dot(V,V))
            return -0.5*np.trace(Conmutator.reshape([shape, shape, shape, shape]), axis1 = 1, axis2 = 3)
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
        for i in tqdm(range(self.prev_steps, self.prev_steps + N-1)):
            self.work.append(self.heat[i] - self.energy[i])
        self.prev_steps += N
        self.stroke_count += 1
    def Driven_evolution(self, bath, bath_hamiltonian, td, dt, V):
        Nd = int(td/dt)
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
        for i in tqdm(range(self.prev_steps + 1, self.prev_steps + 2*Nd + 1)):
            self.work.append(self.heat[i] - self.energy[i])
        self.prev_steps += 2*Nd
    