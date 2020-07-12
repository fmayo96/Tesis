import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm

tf=1
dt = 0.001
N = int(tf/dt)
h = 1.5
bh = 0.75
eps=  np.sqrt(10)
Z=np.exp(bh)+np.exp(-bh)
rho1 = np.zeros([N,2,2],dtype=np.complex)
rho2 = np.zeros([N,4,4],dtype=np.complex)
rho3 = np.zeros((N,8,8),dtype=np.complex)
rho4 = np.zeros((N,16,16),dtype=np.complex)
thermal = [[np.exp(-bh)/Z,0],[0,np.exp(bh)/Z]]
rho1[0] = thermal
rho2[0] = np.kron(thermal,thermal)
rho3[0] = np.kron(np.kron(thermal,thermal),thermal)
rho4[0] = np.kron(thermal,np.kron(thermal,np.kron(thermal,thermal)))
rho1E = thermal
rho2E = np.kron(thermal,thermal)
rho3E = np.kron(np.kron(thermal,thermal),thermal)
rho4E = np.kron(thermal,np.kron(thermal,np.kron(thermal,thermal)))
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]

H1 = (np.eye(2)*(h/2))* sz
H2 = np.kron(H1,Id)+np.kron(Id,H1)
H3 = np.kron(H1,np.kron(Id,Id))+np.kron(Id,np.kron(H1,Id))+np.kron(Id,np.kron(Id,H1))
H4 = np.kron(H1,np.kron(Id,np.kron(Id,Id)))+np.kron(Id,np.kron(H1,np.kron(Id,Id)))+np.kron(Id,np.kron(Id,np.kron(H1,Id)))+np.kron(Id,np.kron(Id,np.kron(Id,H1)))
V1 = eps*(np.kron(su,su) + np.kron(sd,sd))
V2 = 2*eps*(np.kron(su,np.kron(su,np.kron(su,su))) + np.kron(sd,np.kron(sd,np.kron(sd,sd)))+np.kron(su,np.kron(sd,np.kron(su,sd))) + np.kron(sd,np.kron(su,np.kron(sd,su))))
V3 = 3*eps*(np.kron(su,np.kron(su,np.kron(su,np.kron(su,np.kron(su,su))))) + np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,sd))))) + np.kron(su,np.kron(sd,np.kron(su,np.kron(su,np.kron(sd,su))))) + np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,np.kron(su,sd)))))+ np.kron(su,np.kron(su,np.kron(sd,np.kron(su,np.kron(su,sd))))) + np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,su)))))+ np.kron(sd,np.kron(su,np.kron(su,np.kron(sd,np.kron(su,su))))) + np.kron(su,np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,sd))))))
V4 = 4*eps*(np.kron(su,np.kron(su,np.kron(su,np.kron(su,np.kron(su,np.kron(su,np.kron(su,su))))))) + np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(sd,sd))))))) + np.kron(su,np.kron(su,np.kron(su,np.kron(sd,np.kron(su,np.kron(su,np.kron(su,sd))))))) + np.kron(sd,np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,np.kron(sd,su)))))))+ np.kron(su,np.kron(su,np.kron(sd,np.kron(su,np.kron(su,np.kron(su,np.kron(sd,su))))))) + np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(su,sd))))))) + np.kron(su,np.kron(sd,np.kron(su,np.kron(su,np.kron(su,np.kron(sd,np.kron(su,su))))))) + np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,sd)))))))+ np.kron(su,np.kron(sd,np.kron(sd,np.kron(sd,np.kron(su,np.kron(sd,np.kron(sd,sd))))))) + np.kron(sd,np.kron(su,np.kron(su,np.kron(su,np.kron(sd,np.kron(su,np.kron(su,su))))))) )
eigval1,eigvec1 = LA.eig(V1)
eigval2,eigvec2 = LA.eig(V2)
eigval3,eigvec3 = LA.eig(V3)
eigval4,eigvec4 = LA.eig(V4)
print(np.max(eigval1))
print(np.max(eigval2)/2)
print(np.max(eigval3)/3)
print(np.max(eigval4)/4)
def D_1(x):
        rho = np.kron(x,rho1E)
        Conmutator = np.dot(V1,np.dot(V1,rho))-np.dot(V1,np.dot(rho,V1))-np.dot(V1,np.dot(rho,V1))+np.dot(rho,np.dot(V1,V1))
        return -0.5*np.trace(np.reshape(Conmutator,[2,2,2,2]), axis1 = 1, axis2 = 3)
def D_2(x):
        rho = np.kron(x,rho2E)
        Conmutator = np.dot(V2,np.dot(V2,rho))-np.dot(V2,np.dot(rho,V2))-np.dot(V2,np.dot(rho,V2))+np.dot(rho,np.dot(V2,V2))
        return -0.5*np.trace(np.reshape(Conmutator,[4,4,4,4]), axis1 = 1, axis2 = 3)

def D_3(x):
        rho = np.kron(x,rho3E)
        Conmutator = np.dot(V3,np.dot(V3,rho))-np.dot(V3,np.dot(rho,V3))-np.dot(V3,np.dot(rho,V3))+np.dot(rho,np.dot(V3,V3))
        return -0.5*np.trace(np.reshape(Conmutator,[8,8,8,8]), axis1 = 1, axis2 = 3)

def D_4(x):
        rho = np.kron(x,rho4E)
        Conmutator = np.dot(V4,np.dot(V4,rho))-np.dot(V4,np.dot(rho,V4))-np.dot(V4,np.dot(rho,V4))+np.dot(rho,np.dot(V4,V4))
        return -0.5*np.trace(np.reshape(Conmutator,[16,16,16,16]), axis1 = 1, axis2 = 3)

print(np.diag(H4))
def L_1(x):
    return (-1j*(np.dot(H1,rho1[i]) - np.dot(rho1[i],H1)) + D_1(rho1[i]))
def L_2(x):
    return (-1j*(np.dot(H2,rho2[i]) - np.dot(rho2[i],H2)) + D_2(rho2[i]))
def L_3(x):
    return (-1j*(np.dot(H3,rho3[i]) - np.dot(rho3[i],H3)) + D_3(rho3[i]))
def L_4(x):
    return (-1j*(np.dot(H4,rho4[i]) - np.dot(rho4[i],H4)) + D_4(rho4[i]))

for i in range(0,N-1):
    rho1[i+1] = rho1[i] + dt * L_1(rho1[i] + (dt/2) * L_1(rho1[i]))

for i in range(0,N-1):
    rho2[i+1] = rho2[i] + dt * L_2(rho2[i] + (dt/2) * L_2(rho2[i]))

for i in range(0,N-1):
    rho3[i+1] = rho3[i] + dt * L_3(rho3[i] + (dt/2) * L_3(rho3[i]))

for i in range(0,N-1):
    rho4[i+1] = rho4[i] + dt * L_4(rho4[i] + (dt/2) * L_4(rho4[i]))



E1 = np.zeros(N)
E2 = np.zeros(N)
E3 = np.zeros(N)
E4 = np.zeros(N)
for i in range(0,N):
    E1[i] = np.trace(np.dot(H1,rho1[i]))
    E2[i] = np.trace(np.dot(H2,rho2[i]))
    E3[i] = np.trace(np.dot(H3,rho3[i]))
    E4[i] = np.trace(np.dot(H4,rho4[i]))
for i in range(1,N):
    E1[i] -= E1[0]
    E2[i] -= E2[0]
    E3[i] -= E3[0]
    E4[i] -= E4[0]
E1[0] = 0
E2[0] = 0
E3[0] = 0
E4[0] = 0

t = np.linspace(0,tf,N)
plt.figure()
plt.plot(t,E1,linewidth = 2)
plt.plot(t,E2/2.0,linewidth = 2)
plt.plot(t,E3/3.0,linewidth = 2)
plt.plot(t,E4/4.0,linewidth = 2)
plt.legend(["1 qubit","2 qubit", "3 qubit","4 qubit"])
plt.ylabel("Energy per qubit")
plt.xlabel("Time")
plt.grid()
plt.show()



tcarga1 = 0
tcarga2 = 0
tcarga3 = 0
tcarga4 = 0

for i in range(0,N):
    if (E1[i]<=E1[N-1]*0.99):
        tcarga1 = i
    else:
         break
print(tcarga1)

for i in range(0,N):
    if (E2[i]<=E2[N-1]*0.99):
        tcarga2 = i
    else:
         break
print(tcarga2)
for i in range(0,N):
    if (E3[i]<=E3[N-1]*0.99):
        tcarga3 = i
    else:
         break
print(tcarga3)
for i in range(0,N):
    if (E4[i]<=E4[N-1]*0.99):
        tcarga4 = i
    else:
         break
print(tcarga4)


P1 = E1[int(tcarga1)]/(tcarga1*dt)
P2 = E2[int(tcarga2)]/(tcarga2*dt)
P3 = E3[int(tcarga3)]/(tcarga3*dt)
P4 = E4[int(tcarga4)]/(tcarga4*dt)
Pot = [P1,P2,P3,P4]
N = [1,2,3,4]
x = np.linspace(1,4,1000)
x2 = np.linspace(1,4,1000)
plt.figure()
plt.plot(x,x**(2),'r--', linewidth = 2.5)
plt.plot(x,x,'r',linewidth = 2.5)
plt.plot(N,Pot/P1,"o")
plt.fill_between(x,x,x**2,facecolor ='salmon' )
plt.legend(["N^2","N","P(N)/P(1)"], fontsize = 14)
plt.xlabel("N", fontsize = 16)
plt.ylabel("Average Power", fontsize = 16)
plt.show()


