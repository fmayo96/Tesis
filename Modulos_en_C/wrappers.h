#ifndef WRAPPERS_H
#define WRAPPERS_H
int wrapper_Open_evolution(double *out_r, double *out_i, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt);
int wrapper_Driven_evolution(double *out_r, double *out_i, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt);
int wraped_Closed_evolution(double *out_r, double *out_i, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, int dim, double tf, double dt);
#endif