import numpy as np
bh = 0.75
Z=np.exp(bh)+np.exp(-bh)
rho = [[0.5-(1/Z**2),0,0,0],[0,1/Z**2,0,0],[0,0,1/Z**2,0],[0,0,0,0.5-(1/Z**2)]]


rhoa = np.trace(np.reshape(rho,[2,2,2,2]), axis1 = 0, axis2 = 2)
rhob = np.trace(np.reshape(rho,[2,2,2,2]), axis1 = 1, axis2 = 3)
print(rho)
print(rhoa)
print(rhob)
