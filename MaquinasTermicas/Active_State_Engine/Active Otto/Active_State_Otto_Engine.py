import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm
from tqdm import tqdm

N_cycles = 2
t_hot= 5
t_cold = 5
tf = (t_cold + t_hot) * N_cycles
dt = 0.001
N_hot = int(t_hot/dt)
N_cold = int(t_cold/dt)
N_cycle = N_hot + N_cold
N = N_cycle * N_cycles
h_hot = 2
h_cold = 1
h = np.zeros(N)
for i in range(N_hot):
    h[i] = h_hot
    h[i + N_cycle] = h_hot
for i in range(N_hot, N_cycle):
        h[i] = h_cold
        h[i + N_cycle] = h_cold
h[N-1] = h_hot 
h[N_cycle - 1] = h_hot
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
sx = [[0,1],[1,0]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]
H_hot = [[h_hot/2.0,0],[0,-h_hot/2.0]]
H_cold = [[h_cold/2.0,0],[0,-h_cold/2.0]]
t = np.linspace(0,tf,N)
H_driven = np.zeros([N,2,2])
for i in range(N):
    H_driven[i] = [[h[i]/2.0, 0],[0,-h[i]/2.0]]
eps=  np.sqrt(5)
V = eps*(np.kron(su,su) + np.kron(sd,sd))

class reservoir():
    def __init__(self, hamiltonian, beta):
        self.hamiltonian = hamiltonian
        self.beta = beta
        self.state = self.Thermal_state()
    def Thermal_state(self):        
        return np.diag(np.diag(np.exp(np.dot(np.eye(2)*(-self.beta), self.hamiltonian))/np.sum(np.diag(np.diag(np.exp(np.dot(np.eye(2) * (-self.beta), self.hamiltonian)))))))
    def Active_state(self):        
        return np.diag(np.diag(np.exp(np.dot(np.eye(2) * self.beta, self.hamiltonian))/np.sum(np.diag(np.diag(np.exp(np.dot(np.eye(2) * self.beta, self.hamiltonian)))))))
cold_bath = reservoir(H_cold, 10)
hot_bath = reservoir(H_hot,1)
class system():
    def __init__(self):
        self.hamiltonian = H_driven
        self.state = np.zeros([N,2,2],dtype=np.complex)
        self.energy = np.zeros(N)
        self.work = np.zeros(N)
        self.heat = np.zeros(N)
s = system()   
s.state[0] = cold_bath.Active_state() 
s.heat[0] = 0

def D(x, y):
        rho = np.kron(x,y)
        Conmutator = np.dot(V,np.dot(V,rho))-np.dot(V,np.dot(rho,V))-np.dot(V,np.dot(rho,V))+np.dot(rho,np.dot(V,V))
        return (-0.5*np.trace(np.reshape(Conmutator,[2,2,2,2]), axis1 = 1, axis2 = 3))

def L(x, y):
    return -1j*(np.dot(s.hamiltonian[i],x) - np.dot(x,s.hamiltonian[i])) + D(x, y)

def K1(x, y):
    return dt * L(x, y)
def K2(x, y):
    return dt * (L(x + 0.5 * K1(x, y),y))
def K3(x, y):
    return dt * (L(x + 0.5 * K2(x, y),y))
def K4(x, y):
    return dt * (L(x + K3(x, y),y))

def RK4(x, y):
    return x + (1.0/6) * (K1(x, y)+2*K2(x, y)+2*K2(x, y)+K4(x, y))

for i in tqdm(range(N_hot)):
    s.state[i+1] = RK4(s.state[i],hot_bath.state)
    rhop = np.kron(s.state[i],hot_bath.state)
    Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),hot_bath.hamiltonian)))-np.dot(V,np.dot(np.kron(np.eye(2),hot_bath.hamiltonian),V))-np.dot(V,np.dot(np.kron(np.eye(2),hot_bath.hamiltonian),V))+np.dot(np.kron(np.eye(2),hot_bath.hamiltonian),np.dot(V,V))
    s.heat[i+1] = s.heat[i] + dt * np.trace(np.dot(rhop, Conmutator)) * 0.5

for i in tqdm(range(N_hot, N_cycle)):
    s.state[i+1] = RK4(s.state[i], cold_bath.state)
    rhop = np.kron(s.state[i],cold_bath.state)
    Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),cold_bath.hamiltonian)))-np.dot(V,np.dot(np.kron(np.eye(2),cold_bath.hamiltonian),V))-np.dot(V,np.dot(np.kron(np.eye(2),cold_bath.hamiltonian),V))+np.dot(np.kron(np.eye(2),cold_bath.hamiltonian),np.dot(V,V))
    s.heat[i+1] = s.heat[i] + dt * np.trace(np.dot(rhop, Conmutator)) * 0.5
s.heat[N_cycle] = 0
for i in tqdm(range(N_cycle, N_cycle + N_hot)):
    s.state[i+1] = RK4(s.state[i],hot_bath.state)
    rhop = np.kron(s.state[i],hot_bath.state)
    Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),hot_bath.hamiltonian)))-np.dot(V,np.dot(np.kron(np.eye(2),hot_bath.hamiltonian),V))-np.dot(V,np.dot(np.kron(np.eye(2),hot_bath.hamiltonian),V))+np.dot(np.kron(np.eye(2),hot_bath.hamiltonian),np.dot(V,V))
    s.heat[i+1] = s.heat[i] + dt * np.trace(np.dot(rhop, Conmutator)) * 0.5

for i in tqdm(range(N_cycle + N_hot, N-1)):
    s.state[i+1] = RK4(s.state[i], cold_bath.state)
    rhop = np.kron(s.state[i],cold_bath.state)
    Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),cold_bath.hamiltonian)))-np.dot(V,np.dot(np.kron(np.eye(2),cold_bath.hamiltonian),V))-np.dot(V,np.dot(np.kron(np.eye(2),cold_bath.hamiltonian),V))+np.dot(np.kron(np.eye(2),cold_bath.hamiltonian),np.dot(V,V))
    s.heat[i+1] = s.heat[i] + dt * np.trace(np.dot(rhop, Conmutator)) * 0.5


for i in tqdm(range(0,N)):
    s.energy[i] = np.trace(np.dot(s.hamiltonian[i],s.state[i]))
for i in tqdm(range(N)):
    s.work[i] = s.heat[i] - s.energy[i]
for i in tqdm(range(1,N)):
    s.work[i] -= s.work[0]
s.work[0] = 0
rho_max = np.ones(N) * s.state[0,0,0]    
rho_min = np.ones(N) * s.state[0,1,1]    

print(f"Eficiencia = { s.work[-1] / s.heat[-N_cold]}")

plt.figure()
plt.plot(t,s.state[:,0,0], linewidth = 2)
plt.plot(t,s.state[:,0,1], linewidth = 2)
plt.plot(t,s.state[:,1,0], linewidth = 2)
plt.plot(t,s.state[:,1,1], linewidth = 2)
plt.legend(["rho_00","rho_01","rho_10","rho_11"])
plt.show()

plt.figure()
plt.plot(t,s.energy, linewidth = 2)
plt.plot(t,s.work,'r', linewidth = 2)
plt.plot(t,s.heat,'C7-.', linewidth=2)
plt.ylabel("Energy", fontsize = 12)
plt.xlabel("Time", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(["Energy", "Work", "Heat"])
plt.show()


plt.figure()
plt.plot(h[:-1],s.energy[:-1], linewidth=2)
plt.xlabel("h", fontsize = 12)
plt.ylabel("E", fontsize=12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize=12)
plt.show()
