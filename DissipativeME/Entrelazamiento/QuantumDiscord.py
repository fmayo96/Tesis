import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

tf=2
dt = 0.0001
N = int(tf/dt)
h = 1.5
bh = 0.75
eps=  np.sqrt(10)
Z=np.exp(bh)+np.exp(-bh)

rho2 = np.zeros([N,4,4],dtype=np.complex)
thermal = [[np.exp(-bh)/Z,0],[0,np.exp(bh)/Z]]
rho2[0] = np.kron(thermal,thermal)

rho2E = np.kron(thermal,thermal)
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]
H1 = (np.eye(2)*(h/2))* sz
H2 = np.kron(H1,Id)+np.kron(Id,H1)
V2 = 2*eps*(np.kron(su,np.kron(su,np.kron(su,su))) + np.kron(sd,np.kron(sd,np.kron(sd,sd)))+np.kron(su,np.kron(sd,np.kron(su,sd))) + np.kron(sd,np.kron(su,np.kron(sd,su))))


def D_2(x):
        rho = np.kron(x,rho2E)
        Conmutator = np.dot(V2,np.dot(V2,rho))-np.dot(V2,np.dot(rho,V2))-np.dot(V2,np.dot(rho,V2))+np.dot(rho,np.dot(V2,V2))
        return (-0.5*np.eye(4))*np.trace(np.reshape(Conmutator,[4,4,4,4]), axis1 = 0, axis2 = 2)
def L_2(x):
    return (-1j*(np.dot(H2,rho2[i]) - np.dot(rho2[i],H2)) + D_2(rho2[i]))
for i in range(0,N-1):
    rho2[i+1] = rho2[i] + dt * L_2(rho2[i] + (dt/2) * L_2(rho2[i]))

tita = np.zeros(20)
phi = np.zeros(20)
for i in range(20):
    tita[i] = np.pi * (i/20.0)
for i in range(20):
    phi[i] = 2 * np.pi * (i/20.0)

def H(x):
    eigval,eigvec = LA.eig(x)
    entropy = np.zeros(len(eigval))
    for i in range(len(eigval)):
        if eigval[i] == 0:
            entropy[i] = 0
        else:
            entropy[i] = -eigval[i] * np.log(eigval[i])
    return np.sum(entropy)

def MinDiscord(x):
    disc = np.zeros(len(tita)*len(phi))
    rhoS = np.trace(np.reshape(x,[2,2,2,2]),axis1 = 0, axis2 = 2)
    rhoA = np.trace(np.reshape(x,[2,2,2,2]),axis1 = 1, axis2 = 3)
    for i in range(0,20):
        PI_1 = [[np.cos(tita[i])**2,np.exp(-1j*phi[i])*np.sin(tita[i])*np.cos(tita[i])],[np.exp(1j*phi[i])*np.sin(tita[i])*np.cos(tita[i]),np.sin(tita[i])**2]]
        PI_2 = [[np.sin(tita[i])**2,-np.exp(-1j*phi[i])*np.sin(tita[i])*np.cos(tita[i])],[-np.exp(1j*phi[i])*np.sin(tita[i])*np.cos(tita[i]),np.cos(tita[i])**2]]
        rhoS_A1 = np.dot(np.kron(Id,PI_1),np.dot(x,np.kron(Id,PI_1)))/np.trace(np.dot(np.kron(Id,PI_1),x))
        rhoS_A2 = np.dot(np.kron(Id,PI_2),np.dot(x,np.kron(Id,PI_2)))/np.trace(np.dot(np.kron(Id,PI_2),x))
        P1 = np.trace(np.dot(np.kron(Id,PI_1),x))
        P2 = np.trace(np.dot(np.kron(Id,PI_2),x))
        disc[i] = H(rhoS)  - P1 * H(rhoS_A1) - P2 * H(rhoS_A2)
    return np.max(disc)
Discord = np.zeros(N)

I = np.zeros(N, dtype = np.complex)
S = np.zeros(4, dtype = np.complex)
S1 = np.zeros(2, dtype = np.complex)
S2 = np.zeros(2, dtype = np.complex)
for i in range(0,int(N/4)):
    rhoa = np.trace(np.reshape(rho2[i],[2,2,2,2]), axis1=0,axis2=2)
    rhob = np.trace(np.reshape(rho2[i],[2,2,2,2]),axis1=1,axis2=3)

    for j in range(0,4):
        e,v = LA.eig(rho2[i])
        if e[j]!=0:
            S[j] = e[j] * np.log(e[j])
        else:
            S[j] = 0
    for j in range(0,2):
        e1,v1 = LA.eig(rhoa)
        if e1[j]!=0:
            S1[j] = e1[j] * np.log(e1[j])
        else:
            S1[j] = 0
    for j in range(0,2):
                e2,v2 = LA.eig(rhob)
                if e2[j]>0:
                    S2[j] = e2[j] * np.log(e2[j])
                else:
                    S2[j] = 0

    I[i] = -np.sum(S1)-np.sum(S2)+np.sum(S)


for i in range(int(N/4)):
    Discord[i] = I[i] - MinDiscord(rho2[i])




t = np.linspace(0,2,N)
plt.figure()
plt.plot(t[0:int(N/4)],Discord[0:int(N/4)],linewidth = 2)
plt.plot(t[0:int(N/4)],I[0:int(N/4)],'r',linewidth = 2)
plt.xlabel("Time", fontsize = 14)
plt.legend(["Quantum Discord","Mutual Information"], fontsize = 12)
plt.show()
