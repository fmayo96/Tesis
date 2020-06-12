import numpy as np
import matplotlib.pyplot as plt

t_hot= 5
t_cold = 5
tf = t_cold + t_hot
dt = 0.001
N_hot = int(t_hot/dt)
N_cold = int(t_cold/dt)
N = N_hot + N_cold
h = 1.5
beta_hot = 0.1
beta_cold = 2
betah_hot = h * beta_hot/2
betah_cold = h * beta_cold/2
eps=  np.sqrt(0.5)
Z_hot=np.exp(betah_hot)+np.exp(-betah_hot)
Z_cold=np.exp(betah_cold)+np.exp(-betah_cold)
rho = np.zeros([N,2,2],dtype=np.complex)
thermal_hot = [[np.exp(-betah_hot)/Z_hot,0],[0,np.exp(betah_hot)/Z_hot]]
thermal_cold = [[np.exp(-betah_cold)/Z_cold,0],[0,np.exp(betah_cold)/Z_cold]]
active_hot = [[np.exp(betah_hot)/Z_hot,0],[0,np.exp(-betah_hot)/Z_hot]] 
active_cold = [[np.exp(betah_cold)/Z_cold,0],[0,np.exp(-betah_cold)/Z_cold]] 
rho[0] = active_cold 
rhoE_hot = thermal_hot
rhoE_cold = thermal_cold
 
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
sx = [[0,1],[1,0]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]
H = [[h/2,0],[0,-h/2]]
t = np.linspace(0,tf,N)
V = eps*(np.kron(su,su) + np.kron(sd,sd))

def D_hot(x):
        rho = np.kron(x,rhoE_hot)
        Conmutator = np.dot(V,np.dot(V,rho))-np.dot(V,np.dot(rho,V))-np.dot(V,np.dot(rho,V))+np.dot(rho,np.dot(V,V))
        return (-0.5*np.trace(np.reshape(Conmutator,[2,2,2,2]), axis1 = 1, axis2 = 3))

def L_hot(x):
    return (-1j*(np.dot(H,x) - np.dot(x,H)) + D_hot(x))

def K1_hot(x):
    return dt * L_hot(x)
def K2_hot(x):
    return dt * (L_hot(x + 0.5 * K1_hot(x)))
def K3_hot(x):
    return dt * (L_hot(x + 0.5 * K2_hot(x)))
def K4_hot(x):
    return dt * (L_hot(x + K3_hot(x)))

def D_cold(x):
        rho = np.kron(x,rhoE_cold)
        Conmutator = np.dot(V,np.dot(V,rho))-np.dot(V,np.dot(rho,V))-np.dot(V,np.dot(rho,V))+np.dot(rho,np.dot(V,V))
        return (-0.5*np.trace(np.reshape(Conmutator,[2,2,2,2]), axis1 = 1, axis2 = 3))

def L_cold(x):
    return (-1j*(np.dot(H,x) - np.dot(x,H)) + D_cold(x))

def K1_cold(x):
    return dt * L_cold(x)
def K2_cold(x):
    return dt * (L_cold(x + 0.5 * K1_cold(x)))
def K3_cold(x):
    return dt * (L_cold(x + 0.5 * K2_cold(x)))
def K4_cold(x):
    return dt * (L_cold(x + K3_cold(x)))

def RK4_hot(x):
    return x + (1.0/6) * (K1_hot(x)+2*K2_hot(x)+2*K2_hot(x)+K4_hot(x))

def RK4_cold(x):
    return x + (1.0/6) * (K1_cold(x)+2*K2_cold(x)+2*K2_cold(x)+K4_cold(x))


for i in range(0,N_hot):
    rho[i+1] = RK4_hot(rho[i])
for i in range(N_hot, N-1):
    rho[i+1] = RK4_cold(rho[i])
E = np.zeros(N)

W = np.zeros(N)
for i in range(N):
    E[i] = np.trace(np.dot(H,rho[i]))
    W[i] = 2* E[i]
for i in range(1,N):
    W[i] -= W[0]
W[0] = 0
plt.figure()
plt.plot(t,rho[:,0,0], linewidth = 2)
plt.plot(t,rho[:,0,1], linewidth = 2)
plt.plot(t,rho[:,1,0], linewidth = 2)
plt.plot(t,rho[:,1,1], linewidth = 2)
plt.legend(["rho_00","rho_01","rho_10","rho_11"])
plt.show()

plt.figure()
plt.plot(t,E, linewidth = 2)
plt.plot(t,W,'r', linewidth = 2)
plt.ylabel("Energy", fontsize = 12)
plt.xlabel("Time", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(["Energy", "Work"])
plt.show()
