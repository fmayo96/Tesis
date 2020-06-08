import numpy as np
import matplotlib.pyplot as plt

tf = 1
dt = 0.0001
N = int(tf/dt)
H = [[1,1],[1,-1]]
rho = np.zeros((N,2,2),dtype=np.complex)
rho[0] =[[0.6,0],[0,0.4]]

for i in range(0,N-1):
    rho[i+1] = rho[i] + dt* (-1j*(np.dot(H,rho[i])-np.dot(rho[i],H)))

E = np.zeros(N,dtype=np.complex)
t = np.linspace(0,tf,N)
for i in range(0,N-1):
    E[i] = np.trace(np.dot(H,rho[i]))
    print(E[i])
plt.figure()
plt.plot(t,np.imag(rho[:,0,0]))
plt.plot(t,np.imag(rho[:,0,1]))
plt.plot(t,np.imag(rho[:,1,0]))
plt.plot(t,np.imag(rho[:,1,1]))
plt.grid()
plt.legend([00,01,10,11])
plt.show()
plt.figure()
plt.plot(t,E)
plt.show()
