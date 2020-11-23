import numpy as np
import matplotlib.pyplot as plt 
from scipy.linalg import expm


def propagador(alpha):
    M = [[-gamma_pl/2.0,1j*h*alpha/2.0,-1j*h*alpha/2.0,gamma_mn/2.0],[1j*h*alpha/2.0,-1j*h*(1-alpha)-eps**2/4.0,0,-1j*h*alpha/2.0],[-1j*h*alpha/2.0,0,1j*h*(1-alpha)-eps**2/4.0,1j*h*alpha/2.0],[gamma_pl/2.0,-1j*h*alpha/2.0,1j*h*alpha/2.0,-gamma_mn/2.0]]
    M = np.dot(np.eye(4)*dt, M)
    prop = expm(M)
    return prop

def derivada_exp(alpha,dalpha):
    exp_old = np.zeros([4,4], np.complex)
    exp_new = np.zeros([4,4], np.complex)
    exp_old = propagador(alpha)
    exp_new = propagador(alpha + dalpha)
    derivada = (exp_new - exp_old) / dalpha
    return derivada
def gradiente(M,dM,state_evol,i):

    for j in range(N-1):
        if (j != i):
            state_evol[j+1] = np.dot(exp_M[j], state_evol[j])
        else:
            dM = derivada_exp(alpha[j], dalpha)
            state_evol[j+1] = np.dot(dM, state_evol[j])
    grad_state = state_evol[N-1,0,0]
    return grad_state
def ascenso_grad(alpha):
    grad_state = np.ones(N)
    count = 0
    while(max(abs(grad_state))>=1e-3):
        for i in range(N):
            grad_state[i] = gradiente(M, dM, state_evol,i)
        alpha += 1*grad_state
        for i in range(N):
            if (alpha[i]<=0):
                alpha[i] =0
                grad_state[i] = 0
            elif (alpha[i] >=1):
                alpha[i] = 1
                grad_state[i] = 0
        count += 1
        
        #alpha[0] = 0

        #alpha[20:N] = 0
    print(f"Number of iterations = {count}")
    return alpha
#-------Def Variables y Hamiltoniano------
tf = 5
dt = 0.1
N = int(tf/dt)
t = np.linspace(0,tf,N)
h = 1.5 
beta = 1
Z = 2*np.cosh(beta*h/2.0)
eps = np.sqrt(0.5)
state = np.array([[np.exp(-beta*h/2.0)/Z,0],[0,np.exp(beta*h/2.0)/Z]])
dalpha = 0.001
gamma_pl = eps**2*state[0,0]
gamma_mn = eps**2*state[1,1]
state_evol = np.zeros([N,4,1], dtype = np.complex)
alpha = np.random.rand(N)
alpha[0] = 0
alpha[N-1] = 0
M = np.zeros([N,4,4], dtype = np.complex)
dM = np.zeros([4,4], dtype = np.complex)
for i in range(N):
    M[i] = [[-gamma_pl/2.0,1j*h*alpha[i]/2.0,-1j*h*alpha[i]/2.0,gamma_mn/2.0],[1j*h*alpha[i]/2.0,-1j*h*(1-alpha[i])-eps**2/4.0,0,-1j*h*alpha[i]/2.0],[-1j*h*alpha[i]/2.0,0,1j*h*(1-alpha[i])-eps**2/4.0,1j*h*alpha[i]/2.0],[gamma_pl/2.0,-1j*h*alpha[i]/2.0,1j*h*alpha[i]/2.0,-gamma_mn/2.0]]
    M[i] = np.dot(np.eye(4)*dt,M[i])
exp_M = np.zeros([N,4,4], dtype = np.complex)
print(M[0])
print(M[10])
print(M[20])
for i in range(N):
    exp_M[i] = expm(M[i])
state = state.reshape(4,1)
state_evol[0] = state
grad_state = np.zeros(N)
print(alpha)
alpha = ascenso_grad(alpha)
print(alpha)

for i in range(N):
    M[i] = [[-gamma_pl/2.0,1j*h*alpha[i]/2.0,-1j*h*alpha[i]/2.0,gamma_mn/2.0],[1j*h*alpha[i]/2.0,-1j*h*(1-alpha[i])-eps**2/4.0,0,-1j*h*alpha[i]/2.0],[-1j*h*alpha[i]/2.0,0,1j*h*(1-alpha[i])-eps**2/4.0,1j*h*alpha[i]/2.0],[gamma_pl/2.0,-1j*h*alpha[i]/2.0,1j*h*alpha[i]/2.0,-gamma_mn/2.0]]
exp_M = np.zeros([N,4,4], dtype = np.complex)
for i in range(N):
    exp_M[i] = expm(M[i]*dt)
state = state.reshape(4,1)
state_evol[0] = state
for i in range(N-1):
    state_evol[i+1] = np.dot(exp_M[i], state_evol[i])
"""


plt.figure()
plt.ylabel('Rho_00', fontsize = 12)
plt.xlabel('Time', fontsize=12)
plt.plot(t,state_evol[:,0], linewidth = 2)
plt.plot(t,state_evol[:,3])
plt.plot(t,state_evol[:,1])
plt.plot(t,state_evol[:,2])

plt.show()
"""