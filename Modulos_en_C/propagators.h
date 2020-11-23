#ifndef PROPAGATORS_H
#define PROPAGATORS_H
void Dissipator(double complex *dissipator, double complex *state, double complex *bath_state, double complex *interaction, int dim);
void Propagator(double complex *propagator, double complex *state, double complex *hamiltonian, double complex *dissipator, int dim);
void Unitary(double complex *propagator, double complex *state, double complex*hamiltonian, int dim);
#endif
