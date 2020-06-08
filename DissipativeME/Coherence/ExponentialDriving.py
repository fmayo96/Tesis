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
alpha = eps**2/(2*np.cosh(bh))
p = 0
Z=np.exp(bh)+np.exp(-bh)
rho = np.zeros([N,2,2],dtype=np.complex)
rho2 = np.zeros([N,2,2],dtype=np.complex)
rho3 = np.zeros([N,2,2],dtype=np.complex)
thermal = [[np.exp(-bh)/Z,0],[0,np.exp(bh)/Z]]
rho[0] = thermal
rho2[0] = thermal
rho3[0] = thermal
rhoE = thermal

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
H2 = [[h/2,0],[0,-h/2]]
H3 = np.zeros([N,2,2])
Q = np.zeros(N)
W = np.zeros(N)
for i in range(2*Nd):
    H3[i] = (h/2)*(np.dot((np.sin(np.pi*t[i]/(8*td))**2)*np.eye(2),sz) + np.dot((np.cos(np.pi*t[i]/(8*td))**2)*np.eye(2),sx))
for i in range(2*Nd,N):
    H3[i] = H2
def D_1(x):
        rho = np.kron(x,rhoE)
        Conmutator = np.dot(V,np.dot(V,rho))-np.dot(V,np.dot(rho,V))-np.dot(V,np.dot(rho,V))+np.dot(rho,np.dot(V,V))
        return (-0.5*np.trace(np.reshape(Conmutator,[2,2,2,2]), axis1 = 1, axis2 = 3))
        #return alpha*(np.exp(bh)*(np.dot(sd,np.dot(x,su))-0.5*(np.dot(su,np.dot(sd,x))+np.dot(x,np.dot(su,sd)))))+alpha*np.exp(-bh)*(np.dot(su,np.dot(x,sd))-0.5*(np.dot(sd,np.dot(su,x))+np.dot(x,np.dot(sd,su))))
def L_1(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])) + D_1(x))
def L_0(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])))

def L_2(x):
    return (-1j*(np.dot(H2,x) - np.dot(x,H2)) + D_1(x))
def L_3(x):
    return (-1j*(np.dot(H3[i],x) - np.dot(x,H3[i])) +  0.5*D_1(x))


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


for i in range(0,(2*Nd)-1,2):
    rho[i+1] = rho[i] + (1.0/6) * (K1_1(rho[i])+2*K2_1(rho[i])+2*K2_1(rho[i])+K4_1(rho[i]))
    rho[i+1] = np.dot((1 - 0.5*p)*np.eye(2),rho[i+1]) + 0.5*p*(np.dot(2/h*H[i],np.dot(rho[i+1],2/h*H[i])))
    rho[i+2] = rho[i+1] + (1.0/6) * (K1_0(rho[i+1])+2*K2_0(rho[i+1])+2*K2_0(rho[i+1])+K4_0(rho[i+1]))
    rhop = np.kron(rho[i],rhoE)
    Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),H2)))-np.dot(V,np.dot(np.kron(np.eye(2),H2),V))-np.dot(V,np.dot(np.kron(np.eye(2),H2),V))+np.dot(np.kron(np.eye(2),H2),np.dot(V,V))
    Q[i+1] = Q[i] + dt * (np.trace(np.dot(rhop,Conmutator))+(dt/2) * np.trace(np.dot(rhop,Conmutator)))
    Q[i+2] = Q[i+1]
for i in range(2*Nd,N-1):
    rho[i+1] = rho[i] + (1.0/6) * (K1_2(rho[i])+2*K2_2(rho[i])+2*K3_2(rho[i])+K4_2(rho[i]))
    rho[i+1] = np.dot((1 - 0.5*p)*np.eye(2),rho[i+1]) + 0.5*p*(np.dot(2/h*H[i],np.dot(rho[i+1],2/h*H[i])))
    rhop = np.kron(rho[i],rhoE)
    Conmutator = np.dot(V,np.dot(V,np.kron(np.eye(2),H2)))-np.dot(V,np.dot(np.kron(np.eye(2),H2),V))-np.dot(V,np.dot(np.kron(np.eye(2),H2),V))+np.dot(np.kron(np.eye(2),H2),np.dot(V,V))
    Q[i+1] = Q[i] + dt * (np.trace(np.dot(rhop,Conmutator))+(dt/2) * np.trace(np.dot(rhop,Conmutator)))
i = np.argmax(abs(rho[:,0,1]))
print(np.argmax(abs(rho[:,0,1])))
print(rho[i])
print(abs(rho[i,0,1]))
for i in range(0,2*Nd):
    rho3[i+1] = rho3[i] + dt * L_3(rho3[i] + (dt/2) * L_3(rho3[i]))

for i in range(2*Nd,N-1):
    rho3[i+1] = rho3[i] + dt * L_2(rho3[i] + (dt/2) * L_2(rho3[i]))

for i in range(0,N-1):
    rho2[i+1] = rho2[i] + dt * L_2(rho2[i] + (dt/2) * L_2(rho2[i]))




E = np.zeros(N)
E3 = np.zeros(N)
for i in range(N):
    E[i] = np.trace(np.dot(rho[i],H[i]))
    E3[i] = np.trace(np.dot(rho3[i],H3[i]))
E2 = np.zeros(N)
for i in range(0,N):
    E2[i] = np.trace(np.dot(rho2[i],H2))
for i in range(1,N):
    E[i] = E[i] - E2[0]
    E3[i] = E3[i] - E2[0]
    E2[i] = E2[i] - E2[0]
E[0] = 0
E2[0] = 0
E3[0] = 0
"""
plt.figure()
plt.plot(t,H[:,0,0],linewidth = 2)
plt.plot(t,H[:,0,1],"r",linewidth = 2)
plt.legend(["sigma z","sigma x"], fontsize = 12)
plt.xlabel("Time", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

"""

np.savetxt("E_exp.txt",E,delimiter = '\n')

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
plt.plot(t,E3,"-.C7",linewidth = 2)
plt.plot(t,E2,"r--",linewidth = 2)
plt.legend(["Exp H","Sin H","Constant H"], fontsize = 12)
plt.xlabel("Time", fontsize = 14)
plt.ylabel("Energy", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

fig, ax1 = plt.subplots()
left, bottom, width, height = [0.45, 0.33, 0.4, 0.4]
plt.ylabel("Energy", fontsize = 12)
plt.xlabel("Time",fontsize = 12)

ax2 = fig.add_axes([left, bottom, width, height])
ax1.plot(t,E,linewidth = 2)
ax1.plot(t,E3,"-.C7",linewidth = 2)
ax1.plot(t,E2,"r--",linewidth = 2)
ax1.legend(["Exp H","Sin H","Constant H"], fontsize = 11)
ax2.plot(t[1900:4000],E[1900:4000],linewidth = 2)
ax2.plot(t[1900:4000],E3[1900:4000],"-.C7",linewidth = 2)
ax2.plot(t[1900:4000],E2[1900:4000],"r--",linewidth = 2)

plt.show()

plt.figure()
plt.plot(t,E,linewidth = 2)
plt.plot(t,Q, 'C7', linewidth = 2)
plt.plot(t,E-Q, 'r', linewidth = 2)
plt.plot(t,E2,"-.C0",linewidth = 2)
plt.plot(t,-E2, '-.C7', linewidth = 2)
plt.plot(t,2*E2, '-.r', linewidth = 2)
plt.legend(["E driven","Q driven","W driven","E constant","Q constant","W constan"], loc = [0.7,0.2], fontsize = 11)
plt.xlabel("Time", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()
