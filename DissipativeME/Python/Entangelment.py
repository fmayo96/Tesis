import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

tf=5
dt = 0.0001
N = int(tf/dt)
h = 1.5
alpha = 3.86194839
bh = 0.75
Z=np.exp(bh)+np.exp(-bh)
rho = np.zeros((N,4,4),dtype=np.complex)
rho[0] = [[(np.exp(-bh)/Z)**2,0,0,0],[0,1/Z**2,0,0],[0,0,1/Z**2,0],[0,0,0,(np.exp(bh)/Z)**2]]
sz1 = ([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])
sz2 = ([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]])
sup1 = ([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]])
sup2 = ([[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,1,0]])
sdown1 = ([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
sdown2 = ([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
H= [[h,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-h]]
V = [[0,0,0,0],[0,0,10,0],[0,10,0,0],[0,0,0,0]]

for i in range(0,N-1):
    rho[i+1] = rho[i] + dt * (-1j*(np.dot(H,rho[i])-np.dot(rho[i],H)+np.dot(V,rho[i])-np.dot(rho[i],V))+alpha*(np.exp(bh)*(np.dot(sdown1,np.dot(rho[i],sup1))-0.5*(np.dot(sup1,np.dot(sdown1,rho[i]))+np.dot(rho[i],np.dot(sup1,sdown1)))))+alpha*np.exp(-bh)*(np.dot(sup1,np.dot(rho[i],sdown1))-0.5*(np.dot(sdown1,np.dot(sup1,rho[i]))+np.dot(rho[i],np.dot(sdown1,sup1)))))
I = np.zeros(N, dtype = np.complex)
S = np.zeros(4, dtype = np.complex)
S1 = np.zeros(2, dtype = np.complex)
S2 = np.zeros(2, dtype = np.complex)
E = np.zeros(N, dtype = np.complex)
for i in range(0,N):
    rho1 = [[rho[i,0,0]+rho[i,1,1],rho[i,0,2]+rho[i,1,3]],[rho[i,2,0]+rho[i,3,1],rho[i,2,2]+rho[i,3,3]]]
    rho2 = [[rho[i,0,0]+rho[i,2,2],rho[i,0,1]+rho[i,2,3]],[rho[i,1,0]+rho[i,3,2],rho[i,1,1]+rho[i,3,3]]]
    E[i] = np.trace(np.dot(H,rho[i]))
    for j in range(0,4):
        e,v = LA.eig(rho[i])
        if e[j]!=0:
            S[j] = e[j] * np.log(e[j])
        else:
            S[j] = 0
    for j in range(0,2):
        e1,v1 = LA.eig(rho1)
        if e1[j]!=0:
            S1[j] = e1[j] * np.log(e1[j])
        else:
            S1[j] = 0
    for j in range(0,2):
                e2,v2 = LA.eig(rho2)
                if e2[j]>0:
                    S2[j] = e2[j] * np.log(e2[j])
                else:
                    S2[j] = 0

    I[i] = -np.sum(S1)-np.sum(S2)+np.sum(S)
print(np.max(I))
t = np.linspace(0,tf,N)
plt.figure()
plt.plot(t,E)
plt.plot(t,I)
plt.legend(["E","I"])
plt.grid()
plt.show()
