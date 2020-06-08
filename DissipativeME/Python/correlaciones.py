import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm

tf=1
dt = 0.0001
N = int(tf/dt)
h = 1.5
alpha = 3.86194839
bh = 0.75
Z=np.exp(bh)+np.exp(-bh)
rho = np.zeros((N,4,4),dtype=np.complex)
rho2 = np.zeros((N,4,4),dtype=np.complex)
sz1 = ([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])
sz2 = ([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]])
sup1 = ([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]])
sup2 = ([[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,1,0]])
sdown1 = ([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
sdown2 = ([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
rho[0] = [[(np.exp(-bh)/Z)**2,0,0,0],[0,1/Z**2,0,0],[0,0,1/Z**2,0],[0,0,0,(np.exp(bh)/Z)**2]]
rho2[0] = [[(np.exp(-bh)/Z),0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,(np.exp(bh)/Z)]]
H= [[h,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-h]]
V = [[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]]
eq=[[(np.exp(bh)/Z)**2,0,0,0],[0,1/Z**2,0,0],[0,0,1/Z**2,0],[0,0,0,(np.exp(-bh)/Z)**2]]

for i in range(0,N-1):
    rho[i+1] = rho[i] + dt * (-1j*(np.dot(H,rho[i])-np.dot(rho[i],H))+alpha*(np.exp(bh)*(np.dot(sdown1,np.dot(rho[i],sup1))-0.5*(np.dot(sup1,np.dot(sdown1,rho[i]))+np.dot(rho[i],np.dot(sup1,sdown1)))))+alpha*np.exp(-bh)*(np.dot(sup1,np.dot(rho[i],sdown1))-0.5*(np.dot(sdown1,np.dot(sup1,rho[i]))+np.dot(rho[i],np.dot(sdown1,sup1))))+alpha*(np.exp(bh)*(np.dot(sdown2,np.dot(rho[i],sup2))-0.5*(np.dot(sup2,np.dot(sdown2,rho[i]))+np.dot(rho[i],np.dot(sup2,sdown2)))))+alpha*np.exp(-bh)*(np.dot(sup2,np.dot(rho[i],sdown2))-0.5*(np.dot(sdown2,np.dot(sup2,rho[i]))+np.dot(rho[i],np.dot(sdown2,sup2)))))
    rho2[i+1] = rho2[i] + dt * (-1j*(np.dot(H,rho2[i])-np.dot(rho2[i],H))+alpha*(np.exp(bh)*(np.dot(sdown1,np.dot(rho2[i],sup1))-0.5*(np.dot(sup1,np.dot(sdown1,rho2[i]))+np.dot(rho2[i],np.dot(sup1,sdown1)))))+alpha*np.exp(-bh)*(np.dot(sup1,np.dot(rho2[i],sdown1))-0.5*(np.dot(sdown1,np.dot(sup1,rho2[i]))+np.dot(rho2[i],np.dot(sdown1,sup1))))+alpha*(np.exp(bh)*(np.dot(sdown2,np.dot(rho2[i],sup2))-0.5*(np.dot(sup2,np.dot(sdown2,rho2[i]))+np.dot(rho2[i],np.dot(sup2,sdown2)))))+alpha*np.exp(-bh)*(np.dot(sup2,np.dot(rho2[i],sdown2))-0.5*(np.dot(sdown2,np.dot(sup2,rho2[i]))+np.dot(rho2[i],np.dot(sdown2,sup2)))))

Qdot = np.zeros(N,dtype = np.complex)
Q2dot = np.zeros(N,dtype=np.complex)

for i in range(0,N-1):
    Q2dot[i] = -alpha*h*(np.exp(bh)*np.trace(np.dot(sup1,np.dot(sdown1,rho2[i])))-np.exp(-bh)*np.trace(np.dot(sdown1,np.dot(sup1,rho2[i]))))-alpha*h*(np.exp(bh)*np.trace(np.dot(sup2,np.dot(sdown2,rho2[i])))-np.exp(-bh)*np.trace(np.dot(sdown2,np.dot(sup2,rho2[i]))))
    Qdot[i] = -alpha*h*(np.exp(bh)*np.trace(np.dot(sup1,np.dot(sdown1,rho[i])))-np.exp(-bh)*np.trace(np.dot(sdown1,np.dot(sup1,rho[i]))))-alpha*h*(np.exp(bh)*np.trace(np.dot(sup2,np.dot(sdown2,rho[i])))-np.exp(-bh)*np.trace(np.dot(sdown2,np.dot(sup2,rho[i]))))
E = np.zeros(N,dtype = np.complex)
E2 = np.zeros(N,dtype = np.complex)
Fproduct = np.zeros(N,dtype = np.complex)
Fentangled = np.zeros(N,dtype = np.complex)
Dproduct = np.zeros(N,dtype = np.complex)
Dentangled = np.zeros(N,dtype = np.complex)
vproduct = np.zeros(N, dtype=np.complex)
ventangled = np.zeros(N, dtype=np.complex)
Dentangled[0] = 0
Dproduct[0] = 0
for i in range(0,N):
    E[i] = np.trace(np.dot(rho[i],H))
    E2[i] = np.trace(np.dot(rho2[i],H))

for i in range(0,N):
    Fproduct[i] = (np.trace(sqrtm(np.dot(sqrtm(rho[i]),np.dot(eq,sqrtm(rho[i]))))))
    Fentangled[i] = (np.trace(sqrtm(np.dot(sqrtm(rho2[i]),np.dot(eq,sqrtm(rho2[i]))))))
    Dproduct[i] = np.arccos(Fproduct[i])
    Dentangled[i] = np.arccos(Fentangled[i])

t = np.linspace(0,tf,N)


print(E[0])
print(E2[0])
plt.figure()
#plt.plot(t[1:N-1],E[1:N-1])
plt.plot(t[1:N-1],E2[1:N-1])
plt.grid()
plt.show()
plt.figure()
plt.plot(t,Dproduct)
plt.plot(t,Dentangled)
plt.grid()
plt.ylabel("Distance")
plt.legend(["Product", "Entangled"])
plt.show()
