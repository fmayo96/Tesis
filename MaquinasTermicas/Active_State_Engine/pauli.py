import numpy as np 

class pauli():
    def __init__(self):
        self.x = np.array([[0,1],[1,0]])
        self.y = np.array([[0,-1j],[1j, 0]])
        self.z = np.array([[1,0],[0,-1]])
        self.pl = np.array([[0,1],[0,0]])
        self.mn = np.array([[0,0],[1,0]])
        self.eye = np.eye(2)
