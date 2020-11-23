from time_evolution import *
import numpy as np 
from tqdm import tqdm
#from RK4 import RK4

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
    def Open_evolution(self, bath_state, bath_hamiltonian, interaction, tf, dt):
        N = int(tf/dt)
        initial_state = np.zeros([N,self.dim, self.dim], dtype = np.complex)
        initial_state[0] = self.state[-1]
        state = np.zeros([N, self.dim, self.dim], dtype = np.complex)
        state = open_evolution(initial_state, self.hamiltonian[self.prev_steps:], bath_state, bath_hamiltonian, interaction, self.dim, tf, dt)
        for i in range(N):
            self.state.append(state[i])
        for i in tqdm(range(self.prev_steps, self.prev_steps + N - 1)):
            self.energy.append(np.trace(np.dot(self.state[i], self.hamiltonian[i])) - self.energy[0])
            rhop = np.kron(self.state[i], bath_state)
            Conmutator = 0.5 * (np.dot(interaction,np.dot(interaction,np.kron(np.eye(self.dim),bath_hamiltonian)))-np.dot(interaction,np.dot(np.kron(np.eye(self.dim),bath_hamiltonian),interaction))-np.dot(interaction,np.dot(np.kron(np.eye(self.dim),bath_hamiltonian),interaction))+np.dot(np.kron(np.eye(self.dim),bath_hamiltonian),np.dot(interaction,interaction)))
            self.heat.append(self.heat[-1] + dt * np.trace(np.dot(rhop, Conmutator)))
        for i in tqdm(range(self.prev_steps, self.prev_steps + N-1)):
            self.work.append(self.heat[i] - self.energy[i])
        self.prev_steps += N
        self.stroke_count += 1
    def Driven_evolution(self, bath_state, bath_hamiltonian, interaction, td, dt):
        Nd = int(td/dt)
        initial_state = np.zeros([2*Nd,self.dim, self.dim], dtype = np.complex)
        initial_state[0] = self.state[-1]
        state = np.zeros([2*Nd, self.dim, self.dim], dtype = np.complex)
        state = driven_evolution(initial_state, self.hamiltonian, bath_state, bath_hamiltonian, interaction, self.dim, 2*td, dt)
        for i in range(2*Nd):
            self.state.append(state[i])
        for i in tqdm(range(self.prev_steps, self.prev_steps + 2 * Nd)):
            self.energy.append(np.trace(np.dot(self.state[i], self.hamiltonian[i])) - self.energy[0])        
        for i in range(self.prev_steps, self.prev_steps + 2*Nd, 2):
            rhop = np.kron(self.state[i],bath_state)
            Conmutator = 0.5 * (np.dot(interaction,np.dot(interaction,np.kron(np.eye(self.dim),bath_hamiltonian)))-np.dot(interaction,np.dot(np.kron(np.eye(self.dim),bath_hamiltonian),interaction))-np.dot(interaction,np.dot(np.kron(np.eye(self.dim),bath_hamiltonian),interaction))+np.dot(np.kron(np.eye(self.dim),bath_hamiltonian),np.dot(interaction,interaction)))
            self.heat.append(self.heat[-1] + dt * np.trace(np.dot(rhop, Conmutator)))
            self.heat.append(self.heat[-1])
        for i in tqdm(range(self.prev_steps + 1, self.prev_steps + 2*Nd + 1)):
            self.work.append(self.heat[i] - self.energy[i])
        self.prev_steps += 2*Nd
    def Closed_evolution(self, tf, dt):
        N = int(tf/dt)
        initial_state = np.zeros([N,self.dim, self.dim], dtype = np.complex)
        initial_state[0] = self.state[-1]
        state = np.zeros([N, self.dim, self.dim], dtype = np.complex)
        state = closed_evolution(initial_state, self.hamiltonian[self.prev_steps:], self.dim, tf, dt)
        for i in range(N):
            self.state.append(state[i])
        for i in tqdm(range(self.prev_steps, self.prev_steps + N - 1)):
            self.energy.append(np.trace(np.dot(self.state[i], self.hamiltonian[i])) - self.energy[0])            
        for i in tqdm(range(self.prev_steps, self.prev_steps + N-1)):
            self.work.append(-self.energy[i])
        self.prev_steps += N
        self.stroke_count += 1
    
