import numpy as np 
import matplotlib.pyplot as plt 

data = np.loadtxt("test_rho.txt")

rho_00 = data[:,0]
rho_01 = data[:,1]
energy = data[:,2]
plt.figure()
plt.plot(rho_00)
plt.plot(rho_01)
plt.plot(energy)
plt.show()
