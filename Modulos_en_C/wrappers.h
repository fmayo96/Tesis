#ifndef WRAPPERS_H
#define WRAPPERS_H
void wrapper_Open_evolution(double *out_r, double *out_i, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt);
void wrapper_Driven_evolution(double *out_r, double *out_i, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt);
void wrapper_Closed_evolution(double *out_r, double *out_i, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, int dim, double tf, double dt);
void wrapper_energy(double *energy, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, int dim, double tf, double dt);
void wrapper_heat(double *heat, double *state_r, double *state_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt);
void wrapper_heat_driven(double *heat, double *state_r, double *state_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt);
#endif
