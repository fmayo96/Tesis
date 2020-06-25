import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm

tf=20
dt = 0.001
N = int(int(tf/dt)/2)

h_hot = 2
h_cold = 1
b_hot = 1
b_cold = 5
bh_hot = b_hot * h_hot / 2
bh_cold = b_cold * h_cold / 2

eps =  np.sqrt(5)
alpha_hot = eps**2/(2*np.cosh(bh_hot))
Z_hot = np.exp(bh_hot)+np.exp(-bh_hot)
alpha_cold = eps**2/(2*np.cosh(bh_cold))
Z_cold = np.exp(bh_cold)+np.exp(-bh_cold)

rho = np.zeros([2*N,2,2],dtype=np.complex)
thermal_hot = [[np.exp(-bh_hot)/Z_hot,0],[0,np.exp(bh_hot)/Z_hot]]
thermal_cold = [[np.exp(-bh_cold)/Z_cold,0],[0,np.exp(bh_cold)/Z_cold]]
rho[0] = thermal_cold
H_hot = [[h_hot/2,0],[0,-h_hot/2]]
H_cold = [[h_cold/2,0],[0,-h_cold/2]]
Id = [[1,0],[0,1]]
sz = [[1,0],[0,-1]]
sx = [[0,1],[1,0]]
su = [[0,0],[1,0]]
sd = [[0,1],[0,0]]
t = np.linspace(0,tf,2*N)
h_hc = np.zeros(int(N/4))
h_ch = np.zeros(int(N/4))
H_hc = np.zeros([int(N/4),2,2], dtype = complex)
H_ch = np.zeros([int(N/4),2,2], dtype = complex)

for i in range(0,int(N/4)):
    h_hc[i] = h_hot + (h_cold - h_hot) * i / (int(N/4)-1)
    h_ch[i] = h_cold + (h_hot - h_cold) * i / (int(N/4)-1)
    H_hc[i] = [[h_hc[i]/2,0],[0,-h_hc[i]/2]]
    H_ch[i] = [[h_ch[i]/2,0],[0,-h_ch[i]/2]]

H = np.zeros([2*N,2,2],dtype = np.complex)
for i in range(int(N/4)):
    H[i] = H_hot
for i in range(int(N/4), 2*int(N/4)):
    H[i] = H_hc[i-int(N/4)]
for i in range(2*int(N/4), 3*int(N/4)):
    H[i] = H_cold
for i in range(3*int(N/4),N):
    H[i] = H_ch[i-3*int(N/4)]
for i in range(N):
    H[i+N] = H[i]
def D_1(x):
    return alpha_hot*(np.exp(-bh_hot)*(np.dot(sd,np.dot(x,su))-0.5*(np.dot(su,np.dot(sd,x))+np.dot(x,np.dot(su,sd)))))+alpha_hot*np.exp(bh_hot)*(np.dot(su,np.dot(x,sd))-0.5*(np.dot(sd,np.dot(su,x))+np.dot(x,np.dot(sd,su))))
def L_1(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])) + D_1(x))
def K1_1(x):
    return dt * L_1(x)
def K2_1(x):
    return dt * (L_1(x + 0.5 * K1_1(x)))
def K3_1(x):
    return dt * (L_1(x + 0.5 * K2_1(x)))
def K4_1(x):
    return dt * (L_1(x + K3_1(x)))

def L_2(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])))
def K1_2(x):
    return dt * L_2(x)
def K2_2(x):
    return dt * (L_2(x + 0.5 * K1_2(x)))
def K3_2(x):
    return dt * (L_2(x + 0.5 * K2_2(x)))
def K4_2(x):
    return dt * (L_2(x + K3_2(x)))


def D_3(x):
    return alpha_cold*(np.exp(-bh_cold)*(np.dot(sd,np.dot(x,su))-0.5*(np.dot(su,np.dot(sd,x))+np.dot(x,np.dot(su,sd)))))+alpha_cold*np.exp(bh_cold)*(np.dot(su,np.dot(x,sd))-0.5*(np.dot(sd,np.dot(su,x))+np.dot(x,np.dot(sd,su))))
def L_3(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])) + D_3(x))
def K1_3(x):
    return dt * L_3(x)
def K2_3(x):
    return dt * (L_3(x + 0.5 * K1_3(x)))
def K3_3(x):
    return dt * (L_3(x + 0.5 * K2_3(x)))
def K4_3(x):
    return dt * (L_3(x + K3_3(x)))

def L_4(x):
    return (-1j*(np.dot(H[i],x) - np.dot(x,H[i])))
def K1_4(x):
    return dt * L_4(x)
def K2_4(x):
    return dt * (L_4(x + 0.5 * K1_4(x)))
def K3_4(x):
    return dt * (L_4(x + 0.5 * K2_4(x)))
def K4_4(x):
    return dt * (L_4(x + K3_4(x)))


for i in range(0,int(N/4)):
        rho[i+1] = rho[i] + (1.0/6) * (K1_1(rho[i])+2*K2_1(rho[i])+2*K3_1(rho[i])+K4_1(rho[i]))
for i in range(int(N/4)-1,2*int(N/4)):
        rho[i+1] = rho[i] + (1.0/6) * (K1_2(rho[i])+2*K2_2(rho[i])+2*K3_2(rho[i])+K4_2(rho[i]))
for i in range(2*int(N/4)-1,3*int(N/4)):
        rho[i+1] = rho[i] + (1.0/6) * (K1_3(rho[i])+2*K2_3(rho[i])+2*K3_3(rho[i])+K4_3(rho[i]))
for i in range(3*int(N/4)-1,N-1):
        rho[i+1] = rho[i] + (1.0/6) * (K1_4(rho[i])+2*K2_4(rho[i])+2*K3_4(rho[i])+K4_4(rho[i]))
for i in range(N-1,N-1+int(N/4)):
        rho[i+1] = rho[i] + (1.0/6) * (K1_1(rho[i])+2*K2_1(rho[i])+2*K3_1(rho[i])+K4_1(rho[i]))
for i in range(N-1+int(N/4)-1,N-1+2*int(N/4)):
        rho[i+1] = rho[i] + (1.0/6) * (K1_2(rho[i])+2*K2_2(rho[i])+2*K3_2(rho[i])+K4_2(rho[i]))
for i in range(N-1+2*int(N/4)-1,N-1+3*int(N/4)):
        rho[i+1] = rho[i] + (1.0/6) * (K1_3(rho[i])+2*K2_3(rho[i])+2*K3_3(rho[i])+K4_3(rho[i]))
for i in range(N-1+3*int(N/4)-1,N+N-1):
        rho[i+1] = rho[i] + (1.0/6) * (K1_4(rho[i])+2*K2_4(rho[i])+2*K3_4(rho[i])+K4_4(rho[i]))

E = np.zeros(2*N)

for i in range(2*N):
    E[i] = np.trace(np.dot(rho[i],H[i]))

W = (E[2 * N - 1] - E[N + 3 * int(N/4)]) + (E[N + 2 * int(N/4)] - E[N + int(N/4)])
Q_hot = E[N + int(N/4)] - E[N]


Q_otto = -h_hot * (np.sinh(bh_hot)-np.sinh(bh_cold))
W_otto = (h_hot - h_cold)*(np.sinh(bh_hot)-np.sinh(bh_cold))

print("Work = ",W)
print("Heat = ",Q_hot)
print("Otto Heat =", Q_otto)
print("Otto Work =", W_otto)
print("Efficiency = ", abs(W)/Q_hot)
print("Otto Efficiency = ", abs(W_otto)/Q_otto)
h = np.zeros(2*N)
for i in range(int(N/4)):
    h[i] = h_hot
for i in range(int(N/4),2*int(N/4)):
    h[i] = h_hc[i-int(N/4)]
for i in range(2*int(N/4),3*int(N/4)):
    h[i] = h_cold
for i in range(3*int(N/4),N):
    h[i] = h_ch[i-3*int(N/4)]
for i in range(N):
    h[i+N] = h[i]

plt.figure()
plt.plot(t,E, linewidth=2)
plt.show()
"""

plt.figure()
plt.plot(t,rho[:,0,0],linewidth = 2)
plt.plot(t,rho[:,0,1],linewidth = 2)
plt.plot(t,rho[:,1,0],linewidth = 2)
plt.plot(t,rho[:,1,1],linewidth = 2)
plt.legend(["rho 00", "rho 01", "rho 10", "rho 11"], fontsize = 11)
plt.show()

plt.figure()
plt.plot(h[N:2*N],E[N:2*N],linewidth = 2)
plt.xlabel("h", fontsize = 12)
plt.ylabel("Energy", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()
"""