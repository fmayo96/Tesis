import numpy as np
import matplotlib.pyplot as plt

tf = 10
dt = 0.0001
N = int(tf/dt)
rho = np.zeros((N,2,2),dtype=np.complex)
rho[0] = [[1-0.8175,-0.02235725-0.08458327j],[-0.02235725+0.08458327j,0.8175]]
sz=[[1,0],[0,-1]]
sup=[[0,0],[1,0]]
sdown=[[0,1],[0,0]]
h = 1.5
alpha = 0.2730809
bh = 0.75
H=[[0.75,0],[0,-0.75]]

for i in range(0,N-1):
    rho[i+1] = rho[i] + dt * (-1j*(np.dot(H,rho[i])-np.dot(rho[i],H))+alpha*(np.exp(bh)*(np.dot(sdown,np.dot(rho[i],sup))-0.5*(np.dot(sup,np.dot(sdown,rho[i]))+np.dot(rho[i],np.dot(sup,sdown)))))+alpha*np.exp(-bh)*(np.dot(sup,np.dot(rho[i],sdown))-0.5*(np.dot(sdown,np.dot(sup,rho[i]))+np.dot(rho[i],np.dot(sdown,sup)))))

t=np.linspace(0,tf,N)
E =np.zeros(N)
for i in range(0,N-1):
    E[i] = np.trace(np.dot(H,rho[i]))-E[0]

print(E[0])
plt.figure()
plt.plot(t[1:N-1],E[1:N-1])
plt.grid()
plt.show()
plt.figure()
plt.plot(t,rho[:,0,0])
plt.plot(t,rho[:,1,1])
plt.plot(t,rho[:,0,1])
plt.plot(t,rho[:,1,0])
plt.show()
