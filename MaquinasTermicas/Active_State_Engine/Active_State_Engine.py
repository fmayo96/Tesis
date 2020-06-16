import numpy as np
import matplotlib.pyplot as plt

t_hot= 5
t_cold = 5
tf = t_cold + t_hot
td = 0.85   
dt = 0.001
N_hot = int(t_hot/dt)
N_cold = int(t_cold/dt)
N = N_hot + N_cold
Nd = int(td / dt)
h = 1.5
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
sx = [[0,1],[1,0]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]
H = [[h/2.0,0],[0,-h/2.0]]
t = np.linspace(0,tf,N)
eps=  np.sqrt(0.5)
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
cold_bath = reservoir(H, 2)
hot_bath = reservoir(H,0.1)
class system():
    def __init__(self):
        self.hamiltonian = H
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
    return (-1j*(np.dot(H,x) - np.dot(x,H)) + D(x, y))

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

for i in range(0,N_hot):
    s.state[i+1] = RK4(s.state[i],hot_bath.state)
    rhop = np.kron(s.state[i],hot_bath.state)
    Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),H)))-np.dot(V,np.dot(np.kron(np.eye(2),H),V))-np.dot(V,np.dot(np.kron(np.eye(2),H),V))+np.dot(np.kron(np.eye(2),H),np.dot(V,V))
    s.heat[i+1] = s.heat[i] + dt * np.trace(np.dot(rhop, Conmutator)) 
for i in range(N_hot, N-1):
    s.state[i+1] = RK4(s.state[i], cold_bath.state)
    rhop = np.kron(s.state[i],cold_bath.state)
    Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),H)))-np.dot(V,np.dot(np.kron(np.eye(2),H),V))-np.dot(V,np.dot(np.kron(np.eye(2),H),V))+np.dot(np.kron(np.eye(2),H),np.dot(V,V))
    s.heat[i+1] = s.heat[i] + dt * np.trace(np.dot(rhop, Conmutator)) 
for i in range(N):
    s.energy[i] = np.trace(np.dot(s.hamiltonian,s.state[i]))
for i in range(N):
    s.work[i] = s.energy[i] - s.heat[i]
for i in range(1,N):
    s.work[i] -= s.work[0]
s.work[0] = 0    
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
plt.ylabel("Energy", fontsize = 12)
plt.xlabel("Time", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(["Energy", "Work"])
plt.show()
