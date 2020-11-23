import numpy as np 
import matplotlib.pyplot as plt 

eff_3_8 = []
eff_dr_3_8 = []
with open('eff_3-8.txt', 'r') as f:
    for line in f:
        eff_3_8.append(line)
eff_3_8 = np.loadtxt('eff_3-8.txt')
eff_dr_3_8 = np.loadtxt('eff_dr_3-8.txt')
eff_8_13 = np.loadtxt('eff_8-15.txt')
eff_dr_8_13 = np.loadtxt('eff_dr_3-8.txt')
eff_13_18 = np.loadtxt('eff_13-18.txt')
eff_dr_13_18 = np.loadtxt('eff_dr_13-18.txt')
pot_3_8 = np.loadtxt('pot_3-8.txt')
pot_dr_3_8 = np.loadtxt('pot_dr_3-8.txt')
pot_8_13 = np.loadtxt('pot_8-15.txt')
pot_dr_8_13 = np.loadtxt('pot_dr_3-8.txt')
pot_13_18 = np.loadtxt('pot_13-18.txt')
pot_dr_13_18 = np.loadtxt('pot_dr_13-18.txt')

