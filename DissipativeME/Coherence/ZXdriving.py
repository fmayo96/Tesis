import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm

tf=10
td = 0.85
dt = 0.001
N = int(tf/dt)
Nd = int(td/dt)
h = 1.5
bh = 0.75
eps=  np.sqrt(0.5)
Z=np.exp(bh)+np.exp(-bh)
rho = np.zeros([N,2,2],dtype=np.complex)
rho2 = np.zeros([N,2,2],dtype=np.complex)
thermal = [[np.exp(-bh)/Z,0],[0,np.exp(bh)/Z]]
rho[0] = thermal
rho2[0] = thermal
rhoE = thermal

Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
sx = [[0,1],[1,0]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]
t = np.linspace(0,tf,N)
V = eps*(np.kron(su,su) + np.kron(sd,sd))
H = np.zeros([N,2,2])
for i in range(0,2*Nd,2):
    H[i] = (h/2)*(np.dot((np.sin(np.pi*t[i]/(8*td))**2)*np.eye(2),sz) + np.dot((np.cos(np.pi*t[i]/(8*td))**2)*np.eye(2),sx))
for i in range(1,2*Nd,2):
    H[i] = H[i-1]
for i in range(2*Nd,N):
    H[i] = [[h/2,0],[0,-h/2]]
H2 = [[h/2,0],[0,-h/2]]
H3 = np.zeros([N,2,2])
for i in range(2*Nd):
    H3[i] = (h/2)*(np.dot((np.sin(np.pi*t[i]/(8*td))**2)*np.eye(2),sz) + np.dot((np.cos(np.pi*t[i]/(8*td))**2)*np.eye(2),sx))
for i in range(2*Nd,N):
    H3[i] = H2
def D_1(x):
        rho = np.kron(x,rhoE)
        Conmutator = np.dot(V,np.dot(V,rho))-np.dot(V,np.dot(rho,V))-np.dot(V,np.dot(rho,V))+np.dot(rho,np.dot(V,V))
        return (-0.5*np.eye(2))*np.trace(np.reshape(Conmutator,[2,2,2,2]), axis1 = 0, axis2 = 2)
def L_1(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])) + D_1(x))

for i in range(0,(2*Nd)-1,2):
    rho[i+1] = rho[i] + dt * L_1(rho[i] + (dt/2) * L_1(rho[i]))
    rho[i+2] = rho[i+1] + dt * (-1j*(np.dot(H[i],rho[i+1]) - np.dot(rho[i+1],H[i])))
for i in range(2*Nd,N-1):
    rho[i+1] = rho[i] + dt * L_1(rho[i] + (dt/2) * L_1(rho[i]))

def L_2(x):
    return (-1j*(np.dot(H2,x) - np.dot(x,H2)) + D_1(x))
def L_3(x):
    return (-1j*(np.dot(H3[i],x) - np.dot(x,H3[i])) +  0.5*D_1(x))
for i in range(0,2*Nd):
    rho2[i+1] = rho2[i] + dt * L_3(rho2[i] + (dt/2) * L_3(rho2[i]))

for i in range(2*Nd,N-1):
    rho2[i+1] = rho2[i] + dt * L_2(rho2[i] + (dt/2) * L_2(rho2[i]))
"""
for i in range(0,N-1):
    rho2[i+1] = rho2[i] + dt * L_2(rho2[i] + (dt/2) * L_2(rho2[i]))
"""

E = np.zeros(N)
for i in range(N):
    E[i] = np.trace(np.dot(rho[i],H[i]))

E2 = np.zeros(N)
for i in range(0,N):
    E2[i] = np.trace(np.dot(rho2[i],H2))
for i in range(1,N):
    E[i] = E[i] - E2[0]
    E2[i] = E2[i] - E2[0]
E[0] = 0
E2[0] = 0
"""
plt.figure()
plt.plot(t,H3[:,0,0],linewidth = 2)
plt.plot(t,H3[:,0,1],"r",linewidth = 2)
plt.legend(["sigma z","sigma x"], fontsize = 12)
plt.xlabel("Time", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()
"""
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
"""
plt.figure()
plt.plot(t,E,linewidth = 2)
plt.plot(t,E2,"r--",linewidth = 2)
plt.legend(["Driven H","Constant H"], fontsize = 12)
plt.xlabel("Time", fontsize = 14)
plt.ylabel("Energy", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()
"""
