import numpy as np
import ctypes as C

matop = C.CDLL('/Volumes/SeagateFranco/Franco/C/Ctypes_tutorial/./libmymatrixoperations.so')

def partial_trace(M, dim_A, dim_B):
    in1 = np.array(M, dtype=C.c_double)
    out = np.zeros(dim_A*dim_A, dtype=C.c_double)
    intp = C.POINTER(C.c_double)
    i = in1.ctypes.data_as(intp)
    j = out.ctypes.data_as(intp)
    matop.partial_trace(j, i, dim_A, dim_B)
    c = np.zeros([dim_A, dim_A])
    for i in range(dim_A):
        for j in range(dim_A):
            c[i,j] = out[i*dim_A + j]
    return c

