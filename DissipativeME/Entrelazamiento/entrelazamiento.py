import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm

tf=1
dt = 0.0001
N = int(tf/dt)
h = 1.5
bh = 0.75
eps=  np.sqrt(10)
alpha = 3.86194839
Z=np.exp(bh)+np.exp(-bh)
rho = np.zeros([N,4,4],dtype=np.complex)
rho2 = np.zeros([N,4,4],dtype=np.complex)
rho3 = np.zeros((N,8,8),dtype=np.complex)
thermal = [[np.exp(-bh)/Z,0],[0,np.exp(bh)/Z]]
rho[0] = np.kron(thermal,thermal)
rho2[0] = np.kron(thermal,thermal)
rho3[0] = np.kron(np.kron(thermal,thermal),thermal)
Sz = [[1,0],[0,-1]]
Su = [[0,0],[1,0]]
Sd = [[0,1],[0,0]]
Id = [[1,0],[0,1]]
H= [[h,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-h]]
H3 = np.dot((h/2)*np.eye(8),np.kron(Sz,np.kron(Id,Id)))+np.dot((h/2)*np.eye(8),np.kron(Id,np.kron(Sz,Id)))+np.dot((h/2)*np.eye(8),np.kron(Id,np.kron(Id,Sz)))
Sy = [[0,-1j],[1j,0]]
Sdown12 = [[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
Sup12 = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]]
Supdown12 = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]
Sdownup12 = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
Sup1down2 = [[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]]
Sdown1up2 = [[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]
Suddu = [[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
Sduud = [[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]
sz1 = ([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])
sz2 = ([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]])
sup1 = ([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]])
sup2 = ([[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,1,0]])
sdown1 = ([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
sdown2 = ([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
Su3 = np.kron(Su,np.kron(Su,Su))
Sd3 = np.kron(Sd,np.kron(Sd,Sd))
Sud3 = np.dot(Su3,Sd3)
Sdu3 = np.dot(Sd3,Su3)

Ve = np.dot(2*np.eye(16),np.kron(Su,np.kron(Su,np.kron(Su,Su))) + np.kron(Sd,np.kron(Sd,np.kron(Sd,Sd)))+np.kron(Su,np.kron(Sd,np.kron(Su,Sd)))+np.kron(Sd,np.kron(Su,np.kron(Sd,Su))))
V = np.kron(Su,np.kron(Id,np.kron(Su,Id))) + np.kron(Sd,np.kron(Id,np.kron(Sd,Id)))+ np.kron(Id,np.kron(Su,np.kron(Id,Su)))+ np.kron(Id,np.kron(Sd,np.kron(Id,Sd)))

avale,avece = np.linalg.eig(Ve)
aval,avec = np.linalg.eig(V)
print("Norma Entrelazada : ", avale)
print("Norma separable:", aval)
gammapl = np.trace(np.dot(rho[0],Supdown12))*(4*eps*eps)
gammamin = np.trace(np.dot(rho[0],Sdownup12))*(4*eps*eps)
gammapl3 = np.trace(np.dot(rho3[0],Sud3))*(9*eps*eps)
gammamin3 = np.trace(np.dot(rho3[0],Sdu3))*(9*eps*eps)
def f(x):
    return -1j*(np.dot(H,x)-np.dot(x,H))+gammapl*(-0.5*np.dot(Supdown12,x)-0.5*np.dot(x,Supdown12)+np.dot(Sdown12,np.dot(x,Sup12)))+gammamin*(-0.5*np.dot(Sdownup12,x)-0.5*np.dot(x,Sdownup12)+np.dot(Sup12,np.dot(x,Sdown12)))
def g(x):
    return (-1j*(np.dot(H,x)-np.dot(x,H))+alpha*(np.exp(bh)*(np.dot(sdown1,np.dot(x,sup1))-0.5*(np.dot(sup1,np.dot(sdown1,x))+np.dot(x,np.dot(sup1,sdown1)))))+alpha*np.exp(-bh)*(np.dot(sup1,np.dot(x,sdown1))-0.5*(np.dot(sdown1,np.dot(sup1,x))+np.dot(x,np.dot(sdown1,sup1))))+alpha*(np.exp(bh)*(np.dot(sdown2,np.dot(x,sup2))-0.5*(np.dot(sup2,np.dot(sdown2,x))+np.dot(x,np.dot(sup2,sdown2)))))+alpha*np.exp(-bh)*(np.dot(sup2,np.dot(x,sdown2))-0.5*(np.dot(sdown2,np.dot(sup2,x))+np.dot(x,np.dot(sdown2,sup2)))))
def h(x):
        return -1j*(np.dot(H3,x)-np.dot(x,H3))+gammapl3*(-0.5*np.dot(Sud3,x)-0.5*np.dot(x,Sud3)+np.dot(Sd3,np.dot(x,Su3)))+gammamin3*(-0.5*np.dot(Sdu3,x)-0.5*np.dot(x,Sdu3)+np.dot(Su3,np.dot(x,Sd3)))
for i in range(0,N-1):
    rho[i+1] = rho[i] + dt * f(rho[i] + dt/2 * f(rho[i]))
    rho2[i+1] = rho2[i] + dt * g(rho2[i] + dt/2 * g(rho2[i]))
    rho3[i+1] = rho3[i] + dt * h(rho3[i] + dt/2 * h(rho3[i]))

E = np.zeros(N,dtype = np.complex)
E2 = np.zeros(N,dtype = np.complex)
E3  = np.zeros(N,dtype = np.complex)
R = np.zeros([N,4,4],dtype = np.complex)
rhom = np.zeros([N,4,4],dtype = np.complex)
Concurrence = np.zeros(N,dtype = np.complex)
#E3 = np.zeros(N,dtype = np.complex)
for i in range(0,N):
    E[i] = np.trace(np.dot(rho[i],H))
    E2[i] = np.trace(np.dot(rho2[i],H))
    E3[i] = np.trace(np.dot(rho3[i],H3))
    rhom[i] = np.dot(np.kron(Sy,Sy),np.dot(np.conjugate(rho[i]),np.kron(Sy,Sy)))
    R[i] = sqrtm(np.dot(sqrtm(rho[i]),np.dot(rhom[i],sqrtm(rho[i]))))
    eigenvalues,eigvec = LA.eig(R[i])
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]

    Concurrence[i] = eigenvalues[0]-eigenvalues[1]-eigenvalues[2]-eigenvalues[3]
for i in range(1,N):
    E[i] -= E[0]
    E2[i] -= E2[0]
    E3[i] -= E3[0]
E[0] = 0
E2[0] = 0
E3[0] = 0
Q = np.zeros(N,dtype = np.complex)
Q2 = np.zeros(N,dtype = np.complex)
Qdot = np.zeros(N,dtype = np.complex)
Qdot2 = np.zeros(N,dtype = np.complex)
Q[0] = 0
Q2[0] = 0
"""for i in range(0,N):
    Qdot[i] = -2*h*(gammapl*np.trace(np.dot(rho[0],Supdown12))-gammamin*np.trace(np.dot(rho[0],Sdownup12)))
    Qdot2[i] = -alpha*h*(np.exp(bh)*np.trace(np.dot(sup1,np.dot(sdown1,rho2[i])))-np.exp(-bh)*np.trace(np.dot(sdown1,np.dot(sup1,rho2[i])))+np.exp(bh)*np.trace(np.dot(sup2,np.dot(sdown2,rho2[i])))-np.exp(-bh)*np.trace(np.dot(sdown2,np.dot(sup2,rho2[i]))))
for i in range(0,N-1):
    Q[i+1] = Q[i] + dt*Qdot[i]
    Q2[i+1] = Q2[i] + dt * Qdot2[i]
"""
t = np.linspace(0,tf,N)
for i in range(0,N):
    print(rho[i],file,rho2qubits.txt)
I = np.zeros(N, dtype = np.complex)
S = np.zeros(4, dtype = np.complex)
S1 = np.zeros(2, dtype = np.complex)
S2 = np.zeros(2, dtype = np.complex)
for i in range(0,N):
    rhoa = np.trace(np.reshape(rho[i],[2,2,2,2]), axis1=0,axis2=2)
    rhob = np.trace(np.reshape(rho[i],[2,2,2,2]),axis1=1,axis2=3)

    for j in range(0,4):
        e,v = LA.eig(rho[i])
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
print(R[247])
i=247

rhoa = np.trace(np.reshape(rho[i],[2,2,2,2]), axis1=0,axis2=2)
rhob = np.trace(np.reshape(rho[i],[2,2,2,2]),axis1=1,axis2=3)

print(rhoa)
print(rhob)

I2 = np.zeros(N, dtype = np.complex)
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

    I2[i] = -np.sum(S1)-np.sum(S2)+np.sum(S)

Fproduct = np.zeros(N,dtype = np.complex)
Fentangled = np.zeros(N,dtype = np.complex)
Dproduct = np.zeros(N,dtype = np.complex)
Dentangled = np.zeros(N,dtype = np.complex)
vproduct = np.zeros(N, dtype=np.complex)
ventangled = np.zeros(N, dtype=np.complex)
Dentangled[0] = 0
Dproduct[0] = 0

for i in range(0,N-1):
    Fproduct[i] = (np.trace(sqrtm(np.dot(sqrtm(np.diag(np.diag(rho2[i]))),np.dot(np.diag(np.diag(rho2[i+1])),sqrtm(np.diag(np.diag(rho2[i]))))))))
    Fentangled[i] = (np.trace(sqrtm(np.dot(sqrtm(np.diag(np.diag(rho[i]))),np.dot(np.diag(np.diag(rho[i+1])),sqrtm(np.diag(np.diag(rho[i]))))))))
    vproduct[i] = np.arccos(Fproduct[i])/dt
    ventangled[i] = np.arccos(Fentangled[i])/dt
    Dentangled[i+1] = Dentangled[i] + dt * ventangled[i]
    Dproduct[i+1] = Dproduct[i] + dt * vproduct[i]
"""
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,Dentangled)
plt.plot(t,Dproduct)
plt.legend(["Entrelazado","Separable"])
plt.grid()
plt.ylabel("Distance")
plt.subplot(2,1,2)
plt.plot(t,ventangled)
plt.plot(t,vproduct)
plt.legend(["Entrelazado","Separable"])
plt.grid()
plt.ylabel("Speed")
plt.xlabel("Time")
plt.show()
"""
"""I3 = np.zeros(N, dtype = np.complex)
for i in range(0,N):
    rhoa = [[rho3[i,0,0]+rho3[i,1,1],rho3[i,0,2]+rho3[i,1,3]],[rho3[i,2,0]+rho3[i,3,1],rho3[i,2,2]+rho3[i,3,3]]]
    rhob = [[rho3[i,0,0]+rho3[i,2,2],rho3[i,0,1]+rho3[i,2,3]],[rho3[i,1,0]+rho3[i,3,2],rho3[i,1,1]+rho3[i,3,3]]]

    for j in range(0,4):
        e,v = LA.eig(rho3[i])
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

    I3[i] = -np.sum(S1)-np.sum(S2)+np.sum(S)
"""
print(rho3[0])
print("")
print(rho3[N-1])
plt.figure()
plt.plot(t,E/2)
plt.plot(t,E2/2)
#plt.plot(t,Q)
#plt.plot(t,Q2)
plt.plot(t,E3/3)
plt.ylabel("Energia (por qubit)")
plt.xlabel("Tiempo")
plt.legend(["N = 1", "N = 2", "N = 3"])
plt.grid()
plt.show()
plt.figure()
plt.title("Informacion Mutua")
plt.plot(t,I)
plt.plot(t,I2)
plt.plot(t,Concurrence)
#plt.plot(t,I3)
plt.xlabel("Tiempo")
plt.legend(["Entrelazado", "Separable","Concurrence"])
plt.grid()
plt.show()
plt.figure()
plt.plot(t,rho2[:,0,0])
plt.plot(t,rho2[:,1,1])
plt.plot(t,rho2[:,2,2])
plt.plot(t,rho2[:,3,3])
#plt.plot(t,rho3[:,4,4])
#plt.plot(t,rho3[:,5,5])
#plt.plot(t,rho3[:,6,6])
#plt.plot(t,rho3[:,7,7])

plt.legend(["00","11","22","33"])
plt.grid()
plt.show()
