#ifndef PROPAGATORS_H
#define PROPAGATORS_H
int Dissipator(double complex *dissipator, double complex *state, double complex *bath_state, double complex *interaction, int dim);
int Propagator(double complex *propagator, double complex *state, double complex *hamiltonian, double complex *dissipator, int dim);
int Unitary(double complex *propagator, double complex *state, double complex*hamiltonian, int dim);
#endif
