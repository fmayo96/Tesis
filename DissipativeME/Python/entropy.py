import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
bh = 0.75
Z=np.exp(bh)+np.exp(-bh)

rho = [[(np.exp(-bh)/Z)**2,0,0,0],[0,1/Z**2,0,0],[0,0,1/Z**2,0],[0,0,0,(np.exp(bh)/Z)**2]]
rhoE = [[(np.exp(-bh)/Z),0,0,1/Z],[0,0,0,0],[0,0,0,0],[1/Z,0,0,(np.exp(bh)/Z)]]
eq=[[(np.exp(bh)/Z)**2,0,0,0],[0,1/Z**2,0,0],[0,0,1/Z**2,0],[0,0,0,(np.exp(-bh)/Z)**2]]
rho1 = [[(np.exp(-bh)/Z),0],[0,(np.exp(bh)/Z)]]
rho2 = [[(np.exp(-bh)/Z),0],[0,(np.exp(bh)/Z)]]
S = np.zeros(4)
S1 = np.zeros(2)
S2 = np.zeros(2)
e,v = LA.eig(rho)


Ve = np.zeros([16,16])
Ve[0,15] = 1
Ve[15,0] = 1
aval,avec=np.linalg.eig(Ve)
print(aval)
V = np.zeros([4,4])
V[0,3]=1
V[3,0] = 1
aval2,avec2 = np.linalg.eig(V)
print(aval2)

Fproduct = (np.trace(sqrtm(np.dot(sqrtm(rho),np.dot(eq,sqrtm(rho))))))
Fentangled = (np.trace(sqrtm(np.dot(sqrtm(rhoE),np.dot(eq,sqrtm(rhoE))))))

Dproduct = np.arccos(Fproduct)
Dentangled = np.arccos(Fentangled)

print("Product distance = ", Dproduct)
print("Entangled distance = ", Dentangled)

for i in range(0,4):
    e,v = LA.eig(rhoE)
    if e[i]!=0:
        S[i] = e[i] * np.log(e[i])
    else:
        S[i] = 0
for i in range(0,2):
    e1,v1 = LA.eig(rho1)
    if e1[i]!=0:
        S1[i] = e1[i] * np.log(e1[i])
    else:
        S1[i] = 0
for i in range(0,2):
            e2,v2 = LA.eig(rho2)
            if e2[i]>0:
                S2[i] = e2[i] * np.log(e2[i])
            else:
                S2[i] = 0

print("I= ",-np.sum(S1)-np.sum(S2)+np.sum(S))
