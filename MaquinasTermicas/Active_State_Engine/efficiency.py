import numpy as np
import matplotlib.pyplot as plt

h2 = 1
beta1 = 0.01
h1 = np.linspace(2, 10, 1000)

beta2 = 10
beta1_1 = 0.05
beta1_2 = 0.01
beta1_3 = 0.2

work = (h1[i] - h2)/2.0 * np.tanh(beta2*h2/2.0) - (h1[i] + h2)/2.0 * np.tanh(beta1 * h1[i] / 2.0)
heat = h1[i]/2.0*(np.tanh(beta2*h2/2.0)-np.tanh(beta1*h1[i]/2.0))
eff = work/heat






print(f"Eficiencia = {eff}")
"""
eff = 1-h2/h1*(np.tanh(beta2*h2/2.0)+np.tanh(beta1*h1/2.0)/(np.tanh(beta2*h2/2.0)-np.tanh(beta1*h1/2.0)))
eff2 = 1-h2/h1*(np.tanh(beta2*h2/2.0)+np.tanh(beta1_1*h1/2.0)/(np.tanh(beta2*h2/2.0)-np.tanh(beta1_1*h1/2.0)))
eff3 = 1-h2/h1*(np.tanh(beta2*h2/2.0)+np.tanh(beta1_2*h1/2.0)/(np.tanh(beta2*h2/2.0)-np.tanh(beta1_2*h1/2.0)))
eff4 = 1-h2/h1*(np.tanh(beta2*h2/2.0)+np.tanh(beta1_3*h1/2.0)/(np.tanh(beta2*h2/2.0)-np.tanh(beta1_3*h1/2.0)))
otto = 1-h2/h1
plt.figure()
plt.plot(h1, eff3, linewidth = 2)
plt.plot(h1, eff2, 'C1',linewidth = 2)
plt.plot(h1, eff, 'C2',linewidth = 2)
plt.plot(h1, eff4,'C3', linewidth = 2)
plt.plot(h1, otto,'C7--' ,linewidth=2)
plt.ylabel("Efficiency", fontsize=12)
plt.xlabel('h_hot', fontsize=12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.legend(["beta1 = 0.01", "beta1 = 0.05", "beta1 = 0.1", "beta1 = 0.2", "Otto efficiency"], fontsize=11)
plt.show()"""