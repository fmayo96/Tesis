from qutip import *
from tqdm import tqdm
from RK4 import RK4

class Reservoir():
    def __init__(self, hamiltonian, beta):
        self.hamiltonian = hamiltonian
        self.beta = beta
    def Thermal_state(self):        
        bh = Qobj(-self.beta * self.hamiltonian)
        expbh = bh.expm()
        return expbh/expbh.tr() 
    def Active_state(self):        
        bh = Qobj(self.beta * self.hamiltonian)
        expbh = bh.expm()
        return expbh/expbh.tr() 
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
            rho = tensor(self.state[-1], bath)
            Commutator = commutator(V, commutator(V,rho))
            return -0.5*Commutator.ptrace(0)
    def Propagator(self, hamiltonian, state, bath, V):
            return -1j * commutator(hamiltonian, state) + self.Dissipator(bath, V)
    def Unitary(self, hamiltonian, state, bath, V):
            return -1j * commutator(hamiltonian, state)
    def Time_evolution(self, bath, bath_hamiltonian, tf, dt, V):
        N = int(tf/dt)
        for i in tqdm(range(self.prev_steps, self.prev_steps +  N - 1)):
            self.state.append(RK4(self.Propagator, self.hamiltonian[i], self.state[-1], bath, dt, V))
        for i in tqdm(range(self.prev_steps, self.prev_steps + N - 1)):
            self.energy.append((self.state[i] * self.hamiltonian[i]).tr() - self.energy[0])
            rhop = tensor(self.state[i],bath)
            Commutator = 0.5 * commutator(V,commutator(V, tensor(identity(self.dim), bath_hamiltonian)))
            self.heat.append(self.heat[-1] + dt * (rhop * Commutator).tr())
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
            self.energy.append((self.state[i] * self.hamiltonian[i]).tr() - self.energy[0])
        for i in range(self.prev_steps, self.prev_steps + 2*Nd, 2):
            rhop = tensor(self.state[i],bath)
            Commutator = 0.5 * commutator(V,commutator(V, tensor(identity(self.dim), bath_hamiltonian)))
            self.heat.append(self.heat[-1] + dt * (rhop * Commutator).tr())
            self.heat.append(self.heat[-1] + dt * (rhop * Commutator).tr())
        for i in tqdm(range(self.prev_steps, self.prev_steps + 2*Nd)):
            self.work.append(self.heat[i] - self.energy[i])
        self.prev_steps += 2*Nd
