import numpy as np
import matplotlib.pyplot as plt

h2 = 1
beta1 = 0.1
h1 = np.linspace(0, 100, 100000)
beta2 = 10

W = -(h1 + h2) * np.tanh(beta1*h1/2) + (h1 - h2) * np.tanh(beta2*h2/2)
#Q1 = h1*(np.tanh(beta2*h2/2)-np.tanh(beta1*h1/2))
#Q2 = h2*(np.tanh(beta1*h1/2)-np.tanh(beta2*h2/2))
plt.figure()
plt.plot(h1, W, linewidth=2)
#plt.plot(h2, Q1, 'r',linewidth=2)
#plt.plot(h2, Q2, 'C7', linewidth=2)
plt.plot(h1, np.zeros(len(h1)), linewidth=2)
plt.show()

