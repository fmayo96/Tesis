#ifndef ENERGY_H
#define ENERGY_H
void calc_energy(double *energy, double complex *state, double complex *hamiltonian, int dim, double tf, double dt);
void calc_heat(double *heat, double complex *state, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim, double tf, double dt);
void calc_work(double *work, double *energy, double *heat, int N);
void calc_heat_driven(double *heat, double complex *state, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim,double tf, double dt);
#endif