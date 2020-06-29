import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm
from tqdm import tqdm 

tf=10
td = 0.85
dt = 0.001
N = int(tf/dt)
Nd = int(td/dt)
h = 1.5
bh = 0.75
eps=  np.sqrt(0.5)
alpha = eps**2/(2*np.cosh(bh))

Z=np.exp(bh)+np.exp(-bh)
rho = np.zeros([N,2,2],dtype=np.complex)
thermal = [[np.exp(-bh)/Z,0],[0,np.exp(bh)/Z]]
rho[0] = thermal
rhoE = thermal
H2 = [[h/2,0],[0,-h/2]]
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
sx = [[0,1],[1,0]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]
t = np.linspace(0,tf,N)
V = eps*(np.kron(su,su) + np.kron(sd,sd))
H = np.zeros([N,2,2])
tau = 20*td
for i in range(0,2*Nd,2):
    H[i] = (h/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),sz) + np.dot(np.exp(-t[i]/tau)*np.eye(2),sx))
for i in range(1,2*Nd,2):
    H[i] = H[i-1]
for i in range(2*Nd,N):
    H[i] = [[h/2,0],[0,-h/2]]
Q = np.zeros(N)
W = np.zeros(N)

def D_1(x):
    return alpha*(np.exp(bh)*(np.dot(sd,np.dot(x,su))-0.5*(np.dot(su,np.dot(sd,x))+np.dot(x,np.dot(su,sd)))))+alpha*np.exp(-bh)*(np.dot(su,np.dot(x,sd))-0.5*(np.dot(sd,np.dot(su,x))+np.dot(x,np.dot(sd,su))))
def L_1(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])) + D_1(x))

def L_0(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])))

def L_2(x):
    return (-1j*(np.dot(H2,x) - np.dot(x,H2)) + D_1(x))


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

rho_E = np.zeros([N,2,2], dtype = np.complex)
Cb = np.zeros([N,2,2], dtype = np.complex)
for i in range(N):
    Cb[i] = np.eye(2)
    """eigval, Cb[i] = LA.eig(H[i])
    Cb[i].reshape(2,2)
    Cb[i].transpose()  """
for j in tqdm(range(0,Nc)):
    p = j/(10*Nc)
    

    for i in range(0,(2*Nd)-1,2):
        rho[i+1] = rho[i] + (1.0/6) * (K1_1(rho[i])+2*K2_1(rho[i])+2*K2_1(rho[i])+K4_1(rho[i]))
        rho[i+1] = np.dot((1 - 0.5*p)*np.eye(2),(np.dot(LA.inv(Cb[i]),np.dot(rho[i+1],Cb[i])))) + 0.5*p*(np.dot(sz,np.dot(np.dot(LA.inv(Cb[i]),np.dot(rho[i+1],Cb[i])),sz)))
        rho[i + 1] = np.dot(Cb[i], np.dot(rho[i+1],LA.inv(Cb[i])))
        rho[i+2] = rho[i+1] + (1.0/6) * (K1_0(rho[i+1])+2*K2_0(rho[i+1])+2*K2_0(rho[i+1])+K4_0(rho[i+1]))
        rhop = np.kron(rho[i],rhoE)
        Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),H2)))-np.dot(V,np.dot(np.kron(np.eye(2),H2),V))-np.dot(V,np.dot(np.kron(np.eye(2),H2),V))+np.dot(np.kron(np.eye(2),H2),np.dot(V,V))
        Conmutator = 0.5 * Conmutator
        Q[i+1] = Q[i] + dt * (np.trace(np.dot(rhop,Conmutator))+(dt/2) * np.trace(np.dot(rhop,Conmutator))) 
        Q[i+2] = Q[i+1]
    for i in range(2*Nd,N-1):
        rho[i+1] = rho[i] + (1.0/6) * (K1_2(rho[i])+2*K2_2(rho[i])+2*K3_2(rho[i])+K4_2(rho[i]))
        rho[i+1] = np.dot((1 - 0.5*p)*np.eye(2),(np.dot(LA.inv(Cb[i]),np.dot(rho[i+1],Cb[i])))) + 0.5*p*(np.dot(sz,np.dot((np.dot(LA.inv(Cb[i]),np.dot(rho[i+1],Cb[i]))),sz)))  
        rho[i + 1] = np.dot(Cb[i], np.dot(rho[i+1],LA.inv(Cb[i])))
        rhop = np.kron(rho[i],rhoE)
        Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),H2)))-np.dot(V,np.dot(np.kron(np.eye(2),H2),V))-np.dot(V,np.dot(np.kron(np.eye(2),H2),V))+np.dot(np.kron(np.eye(2),H2),np.dot(V,V))
        Conmutator = 0.5 * Conmutator
        Q[i+1] = Q[i] + dt * (np.trace(np.dot(rhop,Conmutator))+(dt/2) * np.trace(np.dot(rhop,Conmutator))) 
    l = np.argmax(abs(rho[:,0,1]))
    C[j] = abs(rho[l,0,1]) + abs(rho[l,1,0])



    E = np.zeros(N)
    E2 = np.trace(np.dot(thermal,H2))
    for i in range(N):
        E[i] = np.trace(np.dot(rho[i],H[i]))

    for i in range(1,N):
        E[i] = E[i] - E2
    E[0] = 0
    imax = 0
    for i in range(N-10):
        if E[i] <= (0.95 * E[N-1]):
            imax = i
        else:
            break
    P[j] = (E[imax]-E[0])/(imax*dt)
    Eff[j] = 1 - (abs(Q[N-1]) / (2 * E[N-1]))
rho2 = np.zeros([N,2,2],dtype=np.complex)
rho2[0] = thermal

for i in range(0,N-1):
        rho2[i+1] = rho2[i] + (1.0/6) * (K1_2(rho2[i])+2*K2_2(rho2[i])+2*K3_2(rho2[i])+K4_2(rho2[i]))

Ez = np.zeros(N)
for i in range(N):
    Ez[i] = np.trace(np.dot(rho2[i],H2))
for i in range(1,N):
    Ez[i] -= Ez[0]
Ez[0] = 0

for i in range(N-10):
    if Ez[i] <= (0.95 * Ez[N-1]):
        imax = i
    else:
        break
Pz = (Ez[imax]-Ez[0])/(imax*dt)
Pzvec = np.ones(Nc)

p = np.zeros(Nc)
for i in range(Nc):
    p[i] = i/Nc

plt.figure()
plt.plot(p, P/Pz, linewidth = 2)
plt.plot(p, Pzvec,"--r", linewidth = 2)
plt.ylabel("Power", fontsize = 12)
plt.xlabel("p", fontsize = 12)
plt.legend(["Driving H","Constant H"],fontsize = 11)
plt.show()

plt.figure()
plt.plot(p,Eff,linewidth = 2)
plt.plot(p,Pzvec * 0.5,'--r', linewidth = 2)
plt.xlabel("p",fontsize = 12)
plt.ylabel("Efficiency", fontsize = 12)
plt.legend(["Driven H", "Constant H"], fontsize = 11)
plt.show()

plt.figure()
plt.plot(t,rho[:,0,0],linewidth = 2)
plt.plot(t,rho[:,1,1],linewidth = 2)
plt.plot(t,rho[:,0,1],linewidth = 2)
plt.plot(t,rho[:,1,0],"-.",linewidth = 2)
plt.legend(["r coherente","r coherente","r coherente","r coherente"], fontsize = 12)
plt.xlabel("Time", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

plt.figure()
plt.plot(t,E,linewidth = 2)
plt.xlabel("Time", fontsize = 14)
plt.ylabel("Energy", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()
