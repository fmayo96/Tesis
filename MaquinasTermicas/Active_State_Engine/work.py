import numpy as np
import matplotlib.pyplot as plt

h2 = 1
beta1 = 0.1
h1 = np.linspace(0, 100, 100000)
beta2 = 10
beta1_1 = 1
beta1_2 = 0.5
beta1_3 = 0.2


W = (h1 - h2)/2.0 * np.tanh(beta2*h2/2) - (h1 + h2) * np.tanh(beta1*h1/2)/2.0
W2 = (h1 - h2)/2.0 * np.tanh(beta2*h2/2) - (h1 + h2) * np.tanh(beta1_1*h1/2)/2.0
W3 = (h1 - h2)/2.0 * np.tanh(beta2*h2/2) - (h1 + h2) * np.tanh(beta1_2*h1/2)/2.0
W4 = (h1 - h2)/2.0 * np.tanh(beta2*h2/2) - (h1 + h2) * np.tanh(beta1_3*h1/2)/2.0
#Q1 = h1*(np.tanh(beta2*h2/2)-np.tanh(beta1*h1/2))
#Q2 = h2*(np.tanh(beta1*h1/2)-np.tanh(beta2*h2/2))
plt.figure()
plt.plot(h1, W, linewidth=2)
plt.plot(h1, W4,'C2', linewidth=2)
plt.plot(h1, W3, linewidth=2)
plt.plot(h1, W2,'r', linewidth=2)
plt.plot(h1, np.zeros(len(h1)),'C7--', linewidth=2)
plt.legend(["beta1 = 0.1", 'beta1 = 0.2', 'beta1 = 0.5','beta1 = 1'], fontsize = 11)
plt.ylabel("Work", fontsize = 12)
plt.xlabel("h1", fontsize = 12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.show()

