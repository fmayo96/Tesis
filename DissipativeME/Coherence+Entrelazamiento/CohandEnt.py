import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

tf=10
td = 0.85
dt = 0.0001
N = int(tf/dt)
Nd = int(td/dt)
h = 1.5
bh = 0.75
eps=  np.sqrt(0.5)
alpha = eps**2/(2*np.cosh(bh))
Z=np.exp(bh)+np.exp(-bh)
rho = np.zeros([N,2,2],dtype = np.complex)
rho2 = np.zeros([N,4,4],dtype=np.complex)
thermal = [[np.exp(-bh)/Z,0],[0,np.exp(bh)/Z]]
rho[0] = thermal
rho2[0] = np.kron(thermal,thermal)

rho2E = np.kron(thermal,thermal)
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
sx = [[0,1],[1,0]]
su = [[0,1],[0,0]]
sd = [[0,0],[1,0]]
su2 = np.kron(su,su)
sd2 = np.kron(sd,sd)
H = np.zeros([N,2,2])
tau = 20*td
t = np.linspace(0,tf,N)
for i in range(0,2*Nd,2):
    H[i] = (h/2)*(np.dot((1-np.exp(-t[i]/tau))*np.eye(2),sz) + np.dot(np.exp(-t[i]/tau)*np.eye(2),sx))
for i in range(1,2*Nd,2):
    H[i] = H[i-1]
for i in range(2*Nd,N):
    H[i] = [[h/2,0],[0,-h/2]]
H2 = np.zeros([N,4,4])
for i in range(N):
    H2[i] = np.kron(H[i],Id)+np.kron(Id,H[i])
V2 = 2 * eps * (np.kron(su2, su2) + np.kron(sd2, sd2))
Hconst = [[h,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-h]]
def D_1(x):
        rho = np.kron(x,rho2E)
        Conmutator = np.dot(V2,np.dot(V2,rho))-np.dot(V2,np.dot(rho,V2))-np.dot(V2,np.dot(rho,V2))+np.dot(rho,np.dot(V2,V2))
        return (-0.5*np.trace(Conmutator.reshape([4,4,4,4]), axis1 = 1, axis2 = 3))


def L_1(x):
    return (-1j*(np.dot(H2[i],x) - np.dot(x,H2[i])) + D_1(x))
def L_0(x):
    return (-1j*(np.dot(H2[i],x) - np.dot(x,H2[i])))
def L_2(x):
    return (-1j*(np.dot(Hconst,x) - np.dot(x,Hconst)) + D_1(x))

def K1_1(x):
    return dt * L_1(x)
def K2_1(x):
    return dt * (L_1(x + 0.5 * K1_1(x)))
def K3_1(x):
    return dt * (L_1(x + 0.5 * K2_1(x)))
def K4_1(x):
    return dt * (L_1(x + K3_1(x)))
def K1_0(x):
    return dt * L_0(x)
def K2_0(x):
    return dt * (L_0(x + 0.5 * K1_0(x)))
def K3_0(x):
    return dt * (L_0(x + 0.5 * K2_0(x)))
def K4_0(x):
    return dt * (L_0(x + K3_0(x)))

def K1_2(x):
    return dt * L_2(x)
def K2_2(x):
    return dt * (L_2(x + 0.5 * K1_2(x)))
def K3_2(x):
    return dt * (L_2(x + 0.5 * K2_2(x)))
def K4_2(x):
    return dt * (L_2(x + K3_2(x)))


for i in range(0,(2*Nd)-1,2):
    rho2[i+1] = rho2[i] + (1.0/6) * (K1_1(rho2[i])+2*K2_1(rho2[i])+2*K3_1(rho2[i])+K4_1(rho2[i]))
    rho2[i+2] = rho2[i+1] + (1.0/6) * (K1_0(rho2[i+1])+2*K2_0(rho2[i+1])+2*K2_0(rho2[i+1])+K4_0(rho2[i+1]))

for i in range(2*Nd,N-1):
    rho2[i+1] = rho2[i] + (1.0/6) * (K1_2(rho2[i])+2*K2_2(rho2[i])+2*K3_2(rho2[i])+K4_2(rho2[i]))

print(rho2[N-1],"\n")
print(np.kron(thermal,thermal))




E = np.zeros(N)
E2 = np.loadtxt("E_exp.txt")
E0 = np.trace(np.dot(Hconst,np.kron(thermal,thermal)))
for i in range(N):
    E[i] = np.trace(np.dot(rho2[i],H2[i]))
for i in range(1,N):
    E[i] -=E0
E[0] = 0



t = np.linspace(0,tf,N)
plt.figure()
plt.plot(t,E/2, linewidth = 2)
plt.plot(t,E2,'r', linewidth = 2)
plt.show()
