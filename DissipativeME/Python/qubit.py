import numpy as np
from scipy.integrate import   odeint
import matplotlib.pyplot as plt

tf=10
h = 0.0001
N = int(tf/h)

p11 = np.zeros(N,dtype=np.complex)
p22 = np.zeros(N,dtype=np.complex)
p12 = np.zeros(N,dtype=np.complex)
p21 = np.zeros(N,dtype=np.complex)
z = np.zeros(N)
w = np.zeros(N)
W = np.zeros(N)
p11[0] = 1
p22[0] = 0
p12[0] = 0
p21[0] = 0
w[0] = 0
W[0] = 0
alpha = 0.2730809
bh = 0.75
c = np.ones(N) * p22[0]
c2 = np.ones(N) * p11[0]
for i in range(0,N-1):
    p11[i + 1] = p11[i] + h * (-alpha * np.exp(bh) * p11[i] + alpha * np.exp(-bh) * p22[i])
    p22[i + 1] = p22[i] + h * (alpha * np.exp(bh) * p11[i] - alpha * np.exp(-bh) * p22[i])
    p12[i + 1] = p12[i] + h * (1.5j* p12[i] - 0.5 * alpha * (np.exp(bh)+np.exp(-bh)) * p12[i])
    p21[i + 1] = p21[i] + h * (1.5j* p21[i] - 0.5 * alpha * (np.exp(bh)+np.exp(-bh)) * p21[i])
    w[i + 1] = w[i] + h * (11.58* ((np.exp(-bh) * p22[i]) - (np.exp(bh) * p11[i])))
t = np.linspace(0,tf,N)

for i in range(0,N-1):
    W[i+1] = (w[i+1] - w[i])/h
H=np.zeros(N)
dH=np.zeros(N)
for i in range(0,N):
    H[i] = 0.75*(p11[i]-p22[i])
for i in range(1,N-1):
    H[i]=H[i]-H[0]
    dH[i]=(H[i]-H[i-1])/h
H[0] = 0
plt.figure()

plt.plot(t,p11)
plt.plot(t,p22)
plt.plot(t,p12)
plt.plot(t,p21)

plt.legend(['Pe','Pg'], fontsize = 14)
plt.grid()

plt.subplot(2,1,2)
plt.plot(t[1:N],W[1:N])
plt.plot(t[1:N],-W[1:N]/2)
plt.xlabel('Time',fontsize = 14)
plt.legend(["dW/dt","dQ/dt"])
plt.grid()
plt.savefig("qubit.png")
plt.show()
plt.figure()
plt.grid()
plt.plot(t[1:N-1],H[1:N-1])
#plt.plot(t[1:N-1],w[1:N-1])
#plt.plot(t[1:N-1],W[1:N-1])

plt.show()

plt.figure()
plt.plot(t,p12)
plt.plot(t,p21)
plt.show()
