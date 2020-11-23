#ifndef RK4_H
#define RK4_H
void RK4_open(double complex *propagator, double complex *dissipator, double complex *state, double complex *hamiltonian, double complex *bath_state, double complex *bath_hamiltinian, double complex *interaction, double dt, int dim, int step);
void RK4_closed(double complex *propagator, double complex *state, double complex *hamiltonian, double dt, int dim, int step);
#endif
