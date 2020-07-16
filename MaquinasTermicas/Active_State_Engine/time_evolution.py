import numpy as np 
import ctypes as C 
from numpy.ctypeslib import ndpointer
t_evol = C.CDLL("/Users/francomayo/Franco/Tesis/Modulos_en_C/./libtime_evolution.so")

def open_evolution(state, hamiltonian, bath_state, bath_hamiltonian, interaction, dim, tf, dt):
    N = int(tf/dt)
    state_r = np.real(state)
    state_i = np.imag(state)
    hamiltonian_r = np.real(hamiltonian)
    hamiltonian_i = np.imag(hamiltonian)
    bath_state_r = np.real(bath_state)
    bath_state_i = np.imag(bath_state)
    bath_hamiltonian_r = np.real(bath_state)
    bath_hamiltonian_i = np.imag(bath_state)
    interaction_r = np.real(interaction)
    interaction_i = np.imag(interaction)
    out1 = np.zeros(N*dim*dim, dtype = C.c_double)
    out2 = np.zeros(N*dim*dim, dtype = C.c_double)
    in1 = np.array(state_r, dtype = C.c_double)
    in2 = np.array(state_i, dtype = C.c_double)
    in3 = np.array(hamiltonian_r, dtype = C.c_double)
    in4 = np.array(hamiltonian_i, dtype = C.c_double)
    in5 = np.array(bath_state_r, dtype = C.c_double)
    in6 = np.array(bath_state_i, dtype = C.c_double)
    in7 = np.array(bath_hamiltonian_r, dtype = C.c_double)
    in8 = np.array(bath_hamiltonian_i, dtype = C.c_double)
    in9 = np.array(interaction_r, dtype = C.c_double)
    in10 = np.array(interaction_i, dtype = C.c_double)
    in11 = C.c_int(dim)
    in12 = C.c_double(tf)
    in13 = C.c_double(dt)
    intp = C.POINTER(C.c_double)
    pointout1 = out1.ctypes.data_as(intp)
    pointout2 = out2.ctypes.data_as(intp)
    point1 = in1.ctypes.data_as(intp)
    point2 = in2.ctypes.data_as(intp)
    point3 = in3.ctypes.data_as(intp)
    point4 = in4.ctypes.data_as(intp)
    point5 = in5.ctypes.data_as(intp)
    point6 = in6.ctypes.data_as(intp)
    point7 = in7.ctypes.data_as(intp)
    point8 = in8.ctypes.data_as(intp)
    point9 = in9.ctypes.data_as(intp)
    point10 = in10.ctypes.data_as(intp)
    
    t_evol.wrapper_Open_evolution(pointout1, pointout2, point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, in11, in12, in13)
    c = np.zeros([N,dim,dim], dtype = np.complex)
    for i in range(N):
        for j in range(dim):
            for k in range(dim):
                c[i, j, k] = out1[i*dim*dim + j*dim + k] + out2[i*dim*dim + j*dim + k]*1j
    
    return c

def driven_evolution(state, hamiltonian, bath_state, bath_hamiltonian, interaction, dim, tf, dt):
    N = int(tf/dt)
    state_r = np.real(state)
    state_i = np.imag(state)
    hamiltonian_r = np.real(hamiltonian)
    hamiltonian_i = np.imag(hamiltonian)
    bath_state_r = np.real(bath_state)
    bath_state_i = np.imag(bath_state)
    bath_hamiltonian_r = np.real(bath_state)
    bath_hamiltonian_i = np.imag(bath_state)
    interaction_r = np.real(interaction)
    interaction_i = np.imag(interaction)
    out1 = np.zeros(N*dim*dim, dtype = C.c_double)
    out2 = np.zeros(N*dim*dim, dtype = C.c_double)
    in1 = np.array(state_r, dtype = C.c_double)
    in2 = np.array(state_i, dtype = C.c_double)
    in3 = np.array(hamiltonian_r, dtype = C.c_double)
    in4 = np.array(hamiltonian_i, dtype = C.c_double)
    in5 = np.array(bath_state_r, dtype = C.c_double)
    in6 = np.array(bath_state_i, dtype = C.c_double)
    in7 = np.array(bath_hamiltonian_r, dtype = C.c_double)
    in8 = np.array(bath_hamiltonian_i, dtype = C.c_double)
    in9 = np.array(interaction_r, dtype = C.c_double)
    in10 = np.array(interaction_i, dtype = C.c_double)
    in11 = C.c_int(dim)
    in12 = C.c_double(tf)
    in13 = C.c_double(dt)
    intp = C.POINTER(C.c_double)
    pointout1 = out1.ctypes.data_as(intp)
    pointout2 = out2.ctypes.data_as(intp)
    point1 = in1.ctypes.data_as(intp)
    point2 = in2.ctypes.data_as(intp)
    point3 = in3.ctypes.data_as(intp)
    point4 = in4.ctypes.data_as(intp)
    point5 = in5.ctypes.data_as(intp)
    point6 = in6.ctypes.data_as(intp)
    point7 = in7.ctypes.data_as(intp)
    point8 = in8.ctypes.data_as(intp)
    point9 = in9.ctypes.data_as(intp)
    point10 = in10.ctypes.data_as(intp)
    
    t_evol.wrapper_Driven_evolution(pointout1, pointout2, point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, in11, in12, in13)
    c = np.zeros([N,dim,dim], dtype = np.complex)
    for i in range(N):
        for j in range(dim):
            for k in range(dim):
                c[i, j, k] = out1[i*dim*dim + j*dim + k] + out2[i*dim*dim + j*dim + k]*1j
    
    return c
