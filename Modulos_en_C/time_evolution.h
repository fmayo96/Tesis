#ifndef TIME_EVOLUTION_H
#define TIME_EVOLUTION_H
int Open_evolution(double complex *state, double complex *hamiltonian, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim, double tf, double dt);
int Driven_evolution(double complex *state, double complex *hamiltonian, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim, double tf, double dt);
int Closed_evolution(double complex *state, double complex *hamiltonian, int dim, double tf, double dt);
#endif