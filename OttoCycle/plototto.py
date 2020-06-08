import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("test_otto.txt")
t = data[:,0]
x = data[:,1]
y = data[:,2]
N = len(t)
h_hot = 2
h_cold = 1

plt.figure()
plt.plot(t,x, linewidth = 2)
plt.plot(t,y,'r', linewidth = 2)
plt.xlabel("Time", fontsize = 12)
plt.legend(["R_00","R_11"], fontsize = 11)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.show()


h = np.zeros(len(t))
E = np.zeros(len(t))
N_cycles = 6
t_cycle = int(len(t)/N_cycles)

for i in range(0, int(t_cycle/2)):
    h[i] = h_hot
    E[i] = h_hot * (x[i] - y[i])

for i in range(int(t_cycle/2), 2 * int(t_cycle/2)):
    h[i] = h_cold
    E[i] = h_cold * (x[i] - y[i])

for i in range(2 * int(t_cycle/2), 3 *int(t_cycle/2)):
    h[i] = h_hot
    E[i] = h_hot * (x[i] - y[i])
for i in range(3 * int(t_cycle/2), 4 *int(t_cycle/2)):
    h[i] = h_cold
    E[i] = h_cold * (x[i] - y[i])
for i in range(4 * int(t_cycle/2), 5 *int(t_cycle/2)):
    h[i] = h_hot
    E[i] = h_hot * (x[i] - y[i])
for i in range(5 * int(t_cycle/2), 6 *int(t_cycle/2)):
    h[i] = h_cold
    E[i] = h_cold * (x[i] - y[i])
for i in range(6 * int(t_cycle/2), 7 *int(t_cycle/2)):
    h[i] = h_hot
    E[i] = h_hot * (x[i] - y[i])
for i in range(7 * int(t_cycle/2),8 * int(t_cycle/2) ):
    h[i] = h_cold
    E[i] = h_cold * (x[i] - y[i])
for i in range(8 * int(t_cycle/2), 9 *int(t_cycle/2)):
    h[i] = h_hot
    E[i] = h_hot * (x[i] - y[i])
for i in range(9 * int(t_cycle/2),10 * int(t_cycle/2) ):
    h[i] = h_cold
    E[i] = h_cold * (x[i] - y[i])
for i in range(10 * int(t_cycle/2), 11 *int(t_cycle/2)):
    h[i] = h_hot
    E[i] = h_hot * (x[i] - y[i])
for i in range(11 * int(t_cycle/2),12 * int(t_cycle/2) ):
    h[i] = h_cold
    E[i] = h_cold * (x[i] - y[i])



plt.figure()
plt.plot(h[0:int(N/N_cycles)+1],E[0:int(N/N_cycles)+1])
plt.plot(h[int(N/N_cycles):2*int(N/N_cycles)+1],E[int(N/N_cycles):2*int(N/N_cycles)+1])
plt.plot(h[2*int(N/N_cycles):3*int(N/N_cycles)+1],E[2*int(N/N_cycles):3*int(N/N_cycles)+1])
plt.plot(h[3*int(N/N_cycles):4*int(N/N_cycles)+1],E[3*int(N/N_cycles):4*int(N/N_cycles)+1])
plt.plot(h[4*int(N/N_cycles):5*int(N/N_cycles)+1],E[4*int(N/N_cycles):5*int(N/N_cycles)+1])
#plt.plot(h[5*int(N/N_cycles):N],E[5*int(N/N_cycles):N])
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("h", fontsize =12)
plt.ylabel("E", fontsize = 12)
plt.legend(["1º Cycle", "2º Cycle","3º Cycle","4º Cycle","5º Cycle","6º Cycle"], fontsize = 11)
plt.show()
