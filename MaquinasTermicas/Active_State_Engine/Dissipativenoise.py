import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm
from tqdm import tqdm 
from repeatedinteractions import Reservoir
tf=20
td = 0.85
dt = 0.001
N = int(tf/dt)
Nd = int(td/dt)
h_cold = 1
beta_cold = 10
h_hot = 6
beta_hot = 0.01
eps=  np.sqrt(0.5)
alpha = eps**2/(2*np.cosh(beta_cold*h_cold/2.0))
H_cold = np.array([[h_cold/2,0],[0,-h_cold/2]])
H_hot = np.array([[h_hot/2,0],[0,-h_hot/2]])
rho = np.zeros([N,2,2],dtype=np.complex)
cold_bath = Reservoir(H_cold, beta_cold)
hot_bath = Reservoir(H_hot, beta_hot)
rho[0] = hot_bath.Thermal_state()
rhoE = cold_bath.Thermal_state()
bh = beta_cold*h_cold/2.0
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
sx = [[0,1],[1,0]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]
t = np.linspace(0,tf,N)
V = eps*(np.kron(su,su) + np.kron(sd,sd))
H = np.zeros([N,2,2])
tau = 20*td
H[0] = H_cold
for i in range(1,2*Nd,2):
    H[i] = (h_cold/2.0)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),sz) + np.dot(np.exp(-t[i]/tau)*np.eye(2),sx))
for i in range(2,2*Nd,2):
    H[i] = H[i-1]
for i in range(2*Nd,N):
    H[i] = H_cold
Q = np.zeros(N, dtype = np.complex)
W = np.zeros(N, dtype = np.complex)
Q_hot = h_hot*(np.tanh(beta_cold*h_cold/2.0) - np.tanh(beta_hot*h_hot/2.0))/2.0 
def D_1(x):
    return alpha*(np.exp(bh)*(np.dot(sd,np.dot(x,su))-0.5*(np.dot(su,np.dot(sd,x))+np.dot(x,np.dot(su,sd)))))+alpha*np.exp(-bh)*(np.dot(su,np.dot(x,sd))-0.5*(np.dot(sd,np.dot(su,x))+np.dot(x,np.dot(sd,su))))
def L_1(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])) + D_1(x))

def L_0(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])))

def L_2(x):
    return (-1j*(np.dot(H_cold,x) - np.dot(x,H_cold)) + D_1(x))


def K1_1(x):
    return dt * L_1(x)
def K2_1(x):
    return dt * (L_1(x + 0.5 * K1_1(x)))
def K3_1(x):
    return dt * (L_1(x + 0.5 * K2_1(x)))
def K4_1(x):
    return dt * (L_1(x + K3_1(x)))

def K1_2(x):
    return dt * L_2(x)
def K2_2(x):
    return dt * (L_2(x + 0.5 * K1_2(x)))
def K3_2(x):
    return dt * (L_2(x + 0.5 * K2_2(x)))
def K4_2(x):
    return dt * (L_2(x + K3_2(x)))


def K1_0(x):
    return dt * L_0(x)
def K2_0(x):
    return dt * (L_0(x + 0.5 * K1_0(x)))
def K3_0(x):
    return dt * (L_0(x + 0.5 * K2_0(x)))
def K4_0(x):
    return dt * (L_0(x + K3_0(x)))
Nc = 100
C = np.zeros(Nc)
P = np.zeros(Nc)
Eff = np.zeros(Nc)
i_final = 0
rho_E = np.zeros([N,2,2], dtype = np.complex)
Cb = np.zeros([N,2,2], dtype = np.complex)
for i in range(N):
    eigval, Cb[i] = LA.eig(H[i])
    Cb[i].reshape(2,2)
    Cb[i].transpose()  
for j in tqdm(range(0,Nc)):
    p = j/(2*Nc)  
    for i in range(0,(2*Nd)-1,2):
        rho[i+1] = rho[i] + (1.0/6) * (K1_1(rho[i])+2*K2_1(rho[i])+2*K2_1(rho[i])+K4_1(rho[i]))
        rho[i+1] = np.dot((1 - 0.5*p)*np.eye(2),(np.dot(LA.inv(Cb[i]),np.dot(rho[i+1],Cb[i])))) + 0.5*p*(np.dot(sz,np.dot(np.dot(LA.inv(Cb[i]),np.dot(rho[i+1],Cb[i])),sz)))
        rho[i + 1] = np.dot(Cb[i], np.dot(rho[i+1],LA.inv(Cb[i])))
        rho[i+2] = rho[i+1] + (1.0/6) * (K1_0(rho[i+1])+2*K2_0(rho[i+1])+2*K2_0(rho[i+1])+K4_0(rho[i+1]))
        rhop = np.kron(rho[i],rhoE)
        Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),H_cold)))-np.dot(V,np.dot(np.kron(np.eye(2),H_cold),V))-np.dot(V,np.dot(np.kron(np.eye(2),H_cold),V))+np.dot(np.kron(np.eye(2),H_cold),np.dot(V,V))
        Conmutator = 0.5 * Conmutator
        Q[i+1] = Q[i] + dt * (np.trace(np.dot(rhop,Conmutator))+(dt/2) * np.trace(np.dot(rhop,Conmutator))) 
        Q[i+2] = Q[i+1]
    for i in range(2*Nd,N-1):
        rho[i+1] = rho[i] + (1.0/6) * (K1_2(rho[i])+2*K2_2(rho[i])+2*K3_2(rho[i])+K4_2(rho[i]))
        rho[i+1] = np.dot((1 - 0.5*p)*np.eye(2),(np.dot(LA.inv(Cb[i]),np.dot(rho[i+1],Cb[i])))) + 0.5*p*(np.dot(sz,np.dot((np.dot(LA.inv(Cb[i]),np.dot(rho[i+1],Cb[i]))),sz)))  
        rho[i + 1] = np.dot(Cb[i], np.dot(rho[i+1],LA.inv(Cb[i])))
        rhop = np.kron(rho[i],rhoE)
        Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),H_cold)))-np.dot(V,np.dot(np.kron(np.eye(2),H_cold),V))-np.dot(V,np.dot(np.kron(np.eye(2),H_cold),V))+np.dot(np.kron(np.eye(2),H_cold),np.dot(V,V))
        Conmutator = 0.5 * Conmutator
        Q[i+1] = Q[i] + dt * (np.trace(np.dot(rhop,Conmutator))+(dt/2) * np.trace(np.dot(rhop,Conmutator))) 
    l = np.argmax(abs(rho[:,0,1]))
    C[j] = abs(rho[l,0,1]) + abs(rho[l,1,0])
    E = np.zeros(N)
    E2 = np.trace(np.dot(hot_bath.Thermal_state(),H_cold))
    for i in range(N):
        E[i] = np.trace(np.dot(rho[i],H[i]))
    for i in range(1,N):
        E[i] = E[i] - E[0]
    E[0] = 0
    for i in range(N):
        W[i] = Q[i] - E[i] 
    for i in range(N):
        if W[i] >= 0.95 * W[-1]:
            i_final = i
        else: 
            P[j] = W[i_final]/t[i_final]
            break
    Eff[j] = W[-1] / Q_hot
rho2 = np.zeros([N,2,2],dtype=np.complex)
rho2[0] = hot_bath.Thermal_state()


p = np.zeros(Nc)

for i in range(Nc):
    p[i] = i/(5*Nc)  

P_constant = -0.16335
for i in range(Nc):
    P[i] -= P_constant

plt.figure()
plt.plot(p,P,'o')
plt.xlabel("p",fontsize = 12)
plt.ylabel("Power", fontsize = 12)
plt.legend(["Driven H", "Constant H"], fontsize = 11)
plt.show()

plt.figure()
plt.plot(p,P,'.')
plt.xlabel("p",fontsize = 12)
plt.ylabel("Power", fontsize = 12)
plt.legend(["Driven H", "Constant H"], fontsize = 11)
plt.show()

plt.figure()
plt.plot(p,P,linewidth = 2)
plt.xlabel("p",fontsize = 12)
plt.ylabel("Power", fontsize = 12)
plt.legend(["Driven H", "Constant H"], fontsize = 11)
plt.show()

