#ifndef RK4_H
#define RK4_H
int RK4(double complex *propagator, double complex *dissipator, double complex *state, double complex *hamiltonian, double complex *bath_state, double complex *bath_hamiltinian, double dt, int dim, int step);
#endif
