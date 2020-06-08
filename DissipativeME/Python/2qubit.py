import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm
tf=2
dt = 0.0001
N = int(tf/dt)
h = 1.5
alpha = 3.86194839
bh = 0.75
Z=np.exp(bh)+np.exp(-bh)
rho = np.zeros((N,4,4),dtype=np.complex)
sz1 = ([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])
sz2 = ([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]])
sup1 = ([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]])
sup2 = ([[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,1,0]])
sdown1 = ([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
sdown2 = ([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
rho[0] = [[(np.exp(-bh)/Z)**2,0,0,0],[0,1/Z**2,0,0],[0,0,1/Z**2,0],[0,0,0,(np.exp(bh)/Z)**2]]
H= [[h,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-h]]
V = [[0,0,0,0],[0,0,200,0],[0,200,0,0],[0,0,0,0]]
V2 = [[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]]
#H2=[[(h/2)**2,0,0,0],[0,-(h/2)**2,0,0],[0,0,-(h/2)**2,0],[0,0,0,(h/2)**2]]
"""for i in range(0,N-1):
    rho[i+1] = rho[i] + dt * (-1j*(np.dot(H,rho[i])-np.dot(rho[i],H)+np.dot(V,rho[i])-np.dot(rho[i],V))+alpha*(np.exp(bh)*(np.dot(sdown1,np.dot(rho[i],sup1))-0.5*(np.dot(sup1,np.dot(sdown1,rho[i]))+np.dot(rho[i],np.dot(sup1,sdown1)))))+alpha*np.exp(-bh)*(np.dot(sup1,np.dot(rho[i],sdown1))-0.5*(np.dot(sdown1,np.dot(sup1,rho[i]))+np.dot(rho[i],np.dot(sdown1,sup1)))))
"""
rho2 = np.zeros((N,4,4),dtype=np.complex)
rho2[0] = [[(np.exp(-bh)/Z)**2,0,0,0],[0,1/Z**2,0,0],[0,0,1/Z**2,0],[0,0,0,(np.exp(bh)/Z)**2]]

def f(x):
    return (-1j*(np.dot(H,x)-np.dot(x,H)+np.dot(V2,x)-np.dot(x,V2))+alpha*(np.exp(bh)*(np.dot(sdown1,np.dot(x,sup1))-0.5*(np.dot(sup1,np.dot(sdown1,x))+np.dot(x,np.dot(sup1,sdown1)))))+alpha*np.exp(-bh)*(np.dot(sup1,np.dot(x,sdown1))-0.5*(np.dot(sdown1,np.dot(sup1,x))+np.dot(x,np.dot(sdown1,sup1))))+alpha*(np.exp(bh)*(np.dot(sdown2,np.dot(x,sup2))-0.5*(np.dot(sup2,np.dot(sdown2,x))+np.dot(x,np.dot(sup2,sdown2)))))+alpha*np.exp(-bh)*(np.dot(sup2,np.dot(x,sdown2))-0.5*(np.dot(sdown2,np.dot(sup2,x))+np.dot(x,np.dot(sdown2,sup2)))))

def g(x):
    return (-1j*(np.dot(H,x)-np.dot(x,H))+alpha*(np.exp(bh)*(np.dot(sdown1,np.dot(x,sup1))-0.5*(np.dot(sup1,np.dot(sdown1,x))+np.dot(x,np.dot(sup1,sdown1)))))+alpha*np.exp(-bh)*(np.dot(sup1,np.dot(x,sdown1))-0.5*(np.dot(sdown1,np.dot(sup1,x))+np.dot(x,np.dot(sdown1,sup1))))+alpha*(np.exp(bh)*(np.dot(sdown2,np.dot(x,sup2))-0.5*(np.dot(sup2,np.dot(sdown2,x))+np.dot(x,np.dot(sup2,sdown2)))))+alpha*np.exp(-bh)*(np.dot(sup2,np.dot(x,sdown2))-0.5*(np.dot(sdown2,np.dot(sup2,x))+np.dot(x,np.dot(sdown2,sup2)))))

for i in range(0,N-1):
    rho2[i+1] = rho2[i] + dt * f(rho2[i] + dt/2 * f(rho2[i]))

rho3 = np.zeros((N,4,4),dtype=np.complex)
rho3[0] = [[(np.exp(-bh)/Z)**2,0,0,0],[0,1/Z**2,0,0],[0,0,1/Z**2,0],[0,0,0,(np.exp(bh)/Z)**2]]
for i in range(0,N-1):
    rho3[i+1] = rho3[i] + dt * g(rho3[i] + dt/2 * g(rho3[i]))

rho0 = np.zeros((N,2,2),dtype=np.complex)
rho0[0] = [[1-0.8175,0],[0,0.8175]]

sz=[[1,0],[0,-1]]
sup=[[0,0],[1,0]]
sdown=[[0,1],[0,0]]
"""
for i in range(0,N-1):
    rho0[i+1] = rho0[i] + dt * (-1j*(np.dot(H2,rho0[i])-np.dot(rho0[i],H2))+alpha*(np.exp(bh)*(np.dot(sdown,np.dot(rho0[i],sup))-0.5*(np.dot(sup,np.dot(sdown,rho0[i]))+np.dot(rho0[i],np.dot(sup,sdown)))))+alpha*np.exp(-bh)*(np.dot(sup,np.dot(rho0[i],sdown))-0.5*(np.dot(sdown,np.dot(sup,rho0[i]))+np.dot(rho0[i],np.dot(sdown,sup)))))
"""

Qdot=np.zeros(N,dtype = np.complex)
Q=np.zeros(N,dtype = np.complex)
Q[0] = 0

t=np.linspace(0,tf,N)
E =np.zeros(N,dtype=np.complex)
dE=np.zeros(N,dtype=np.complex)
dE2=np.zeros(N,dtype=np.complex)
E2 =np.zeros(N,dtype=np.complex)
E3 =np.zeros(N,dtype=np.complex)
Q2dot = np.zeros(N,dtype=np.complex)
W = np.zeros(N,dtype=np.complex)
Q2 = np.zeros(N,dtype=np.complex)
Q2[0] = 0
Q3dot = np.zeros(N,dtype=np.complex)
Q3 = np.zeros(N,dtype=np.complex)
Q3[0] = 0
I = np.zeros(N, dtype = np.complex)
S = np.zeros(4, dtype = np.complex)
S1 = np.zeros(2, dtype = np.complex)
S2 = np.zeros(2, dtype = np.complex)
for i in range(0,N):
    rhoa = [[rho2[i,0,0]+rho2[i,1,1],rho2[i,0,2]+rho2[i,1,3]],[rho2[i,2,0]+rho2[i,3,1],rho2[i,2,2]+rho2[i,3,3]]]
    rhob = [[rho2[i,0,0]+rho2[i,2,2],rho2[i,0,1]+rho2[i,2,3]],[rho2[i,1,0]+rho2[i,3,2],rho2[i,1,1]+rho2[i,3,3]]]

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
print(np.max(I))

for i in range(0,N):
    #E[i] = np.trace(np.dot(H2,rho0[i]))
    E2[i] = np.trace(np.dot(H,rho2[i])+np.dot(V2,rho2[i]))
    E3[i] = np.trace(np.dot(H,rho3[i]))
    Q2dot[i] = -alpha*h*(np.exp(bh)*np.trace(np.dot(sup1,np.dot(sdown1,rho2[i])))-np.exp(-bh)*np.trace(np.dot(sdown1,np.dot(sup1,rho2[i])))+np.exp(bh)*np.trace(np.dot(sup2,np.dot(sdown2,rho2[i])))-np.exp(-bh)*np.trace(np.dot(sdown2,np.dot(sup2,rho2[i]))))
    Q3dot[i] = -alpha*h*(np.exp(bh)*np.trace(np.dot(sup1,np.dot(sdown1,rho3[i])))-np.exp(-bh)*np.trace(np.dot(sdown1,np.dot(sup1,rho3[i])))+np.exp(bh)*np.trace(np.dot(sup2,np.dot(sdown2,rho3[i])))-np.exp(-bh)*np.trace(np.dot(sdown2,np.dot(sup2,rho3[i]))))
    Qdot[i] =  -alpha*h*(np.exp(bh)*np.trace(np.dot(sup,np.dot(sdown,rho0[i])))-np.exp(-bh)*np.trace(np.dot(sdown,np.dot(sup,rho0[i]))))
for i in range(1,N):
    E[i] = E[i] - E[0]
    E2[i] = E2[i] - E2[0]
    E3[i] = E3[i] - E3[0]

E[0] = 0
E2[0] = 0
E3[0] = 0

for i in range(0,N-1):
    dE[i+1] = (E[i+1]-E[i])/dt
    dE2[i+1] = (E2[i+1]-E2[i])/dt
    Q2[i+1] = Q2[i] + dt*Q2dot[i]
    Q3[i+1] = Q3[i] + dt*Q3dot[i]
    Q[i+1] = Q[i] + dt*Qdot[i]
W2 = E2 - Q2
W3 = E3 - Q3

"""
plt.figure()
plt.plot(t,rho2[:,0,0])
plt.plot(t,rho2[:,2,2])
plt.plot(t,rho2[:,1,1])
plt.plot(t,rho2[:,3,3])
plt.grid()
plt.show()
"""
step = 1000
print("Sin interaccion")
print("\n")
print(rho3[N-1])
print("\n")
print("Con interaccion")
print("\n")
print(rho2[N-1])
print("Eficiencia =", E2[N-2]/W2[N-2])
plt.figure()
#plt.plot(t[1:N-1],E[1:N-1])
plt.plot(t[::step],E2[::step],"o")
plt.plot(t[::step],E3[::step],"r^")
plt.plot(t[::step],W2[::step],"o")
plt.plot(t[::step],W3[::step],"^")
plt.plot(t[::step],Q2[::step],"om")
plt.plot(t[::step],Q3[::step],"^y")
plt.grid()
plt.xlabel("Time")
plt.legend(["E int","E no int","W int","W no int","Q int", "Q no int"])
plt.show()
plt.figure()
plt.plot(t,I)
plt.grid()
plt.show()
