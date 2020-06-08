import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import sqrtm

Su = [[0,0],[1,0]]
Sd = [[0,1],[0,0]]
Id = [[1,0],[0,1]]

print(np.kron(Su,Id))
print(np.kron(Id,Su))
