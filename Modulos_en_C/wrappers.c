#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "time_evolution.h"
#include "matrixoperations.h"
#include "RK4.h"
#include "propagators.h"
#include "energy.h"
void wrapper_Open_evolution(double *out_r, double *out_i, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt)
{   
    int N = (int)(tf/dt), i;
    double complex *state, *hamiltonian, *bath_state, *bath_hamiltonian, *interaction;
    state = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    hamiltonian = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    bath_state = (double complex*) calloc(dim*dim, sizeof(double complex));
    bath_hamiltonian = (double complex*) calloc(dim*dim, sizeof(double complex));
    interaction = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    for(i = 0; i < N*dim*dim; i++)
    {
        *(state + i) = *(state_r + i) + *(state_i + i)*I;
        *(hamiltonian + i) = *(hamiltonian_r + i) + *(hamiltonian_i + i)*I;
    }
    for(i = 0; i < dim*dim; i++)
    {
        *(bath_state + i) = *(bath_state_r + i) + *(bath_state_i + i)*I;
        *(bath_hamiltonian + i) = *(bath_hamiltonian_r + i) + *(bath_hamiltonian_i + i)*I;
    }
    for(i = 0; i < dim*dim*dim*dim; i++)
    {
        *(interaction + i) = *(interaction_r + i) + *(interaction_i + i)*I;
    }
    Open_evolution(state, hamiltonian, bath_state, bath_hamiltonian, interaction, dim, tf, dt);
    for(i = 0; i < N*dim*dim; i++)
    {
        *(state_r + i) = creal(*(state + i));
        *(state_i + i) = cimag(*(state + i));
        *(out_r + i) = *(state_r + i);
        *(out_i + i) = *(state_i + i);
    }
    free(state);
    free(hamiltonian);
    free(bath_state);
    free(bath_hamiltonian);
    free(interaction);
}
void wrapper_Driven_evolution(double *out_r, double *out_i, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt)
{
    int N = (int)(tf/dt), i;
    double complex *state, *hamiltonian, *bath_state, *bath_hamiltonian, *interaction;
    state = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    hamiltonian = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    bath_state = (double complex*) calloc(dim*dim, sizeof(double complex));
    bath_hamiltonian = (double complex*) calloc(dim*dim, sizeof(double complex));
    interaction = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    for(i = 0; i < N*dim*dim; i++)
    {
        *(state + i) = *(state_r + i) + *(state_i + i)*I;
        *(hamiltonian + i) = *(hamiltonian_r + i) + *(hamiltonian_i + i)*I;
    }
    for(i = 0; i < dim*dim; i++)
    {
        *(bath_state + i) = *(bath_state_r + i) + *(bath_state_i + i)*I;
        *(bath_hamiltonian + i) = *(bath_hamiltonian_r + i) + *(bath_hamiltonian_i + i)*I;
    }
    for(i = 0; i < dim*dim*dim*dim; i++)
    {
        *(interaction + i) = *(interaction_r + i) + *(interaction_i + i)*I;
    }
    Driven_evolution(state, hamiltonian, bath_state, bath_hamiltonian, interaction, dim, tf, dt);
    for(i = 0; i < N*dim*dim; i++)
    {
        *(state_r + i) = creal(*(state + i));
        *(state_i + i) = cimag(*(state + i));
        *(out_r + i) = *(state_r + i);
        *(out_i + i) = *(state_i + i);
    }
    free(state);
    free(hamiltonian);
    free(bath_state);
    free(bath_hamiltonian);
    free(interaction);
}
void wrapper_Closed_evolution(double *out_r, double *out_i, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, int dim, double tf, double dt)
{
    int N = (int)(tf/dt), i;
    double complex *state, *hamiltonian;
    state = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    hamiltonian = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    for(i = 0; i < N*dim*dim; i++)
    {
        *(state + i) = *(state_r + i) + *(state_i + i)*I;
        *(hamiltonian + i) = *(hamiltonian_r + i) + *(hamiltonian_i + i)*I;
    }    
    Closed_evolution(state, hamiltonian, dim, tf, dt);
    for(i = 0; i < N*dim*dim; i++)
    {
        *(state_r + i) = creal(*(state + i));
        *(state_i + i) = cimag(*(state + i));
        *(out_r + i) = *(state_r + i);
        *(out_i + i) = *(state_i + i);
    }
    free(state);
    free(hamiltonian);
}
void wrapper_energy(double *energy, double *state_r, double *state_i, double *hamiltonian_r, double *hamiltonian_i, int dim, double tf, double dt)
{
    int N = (int)(tf/dt), i;
    double complex *state, *hamiltonian;
    state = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    hamiltonian = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    for(i = 0; i < N*dim*dim; i++)
    {
        *(state + i) = *(state_r + i) + *(state_i + i)*I;
        *(hamiltonian + i) = *(hamiltonian_r + i) + *(hamiltonian_i + i)*I;
    }    
    calc_energy(energy, state, hamiltonian, dim, tf, dt);
    free(state);
    free(hamiltonian);
}
void wrapper_heat(double *heat, double *state_r, double *state_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt)
{
    int N = (int)(tf/dt), i;
    double complex *state, *bath_state, *bath_hamiltonian, *interaction;
    state = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    bath_state = (double complex*) calloc(dim*dim, sizeof(double complex));
    bath_hamiltonian = (double complex*) calloc(dim*dim, sizeof(double complex));
    interaction = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    for(i = 0; i < N*dim*dim; i++)
    {
        *(state + i) = *(state_r + i) + *(state_i + i)*I;
    }
    for(i = 0; i < dim*dim; i++)
    {
        *(bath_state + i) = *(bath_state_r + i) + *(bath_state_i + i)*I;
        *(bath_hamiltonian + i) = *(bath_hamiltonian_r + i) + *(bath_hamiltonian_i + i)*I;
    }
    for(i = 0; i < dim*dim*dim*dim; i++)
    {
        *(interaction + i) = *(interaction_r + i) + *(interaction_i + i)*I;
    }
    calc_heat(heat, state, bath_state, bath_hamiltonian, interaction, dim, tf, dt);    
    free(state);
    free(bath_state);
    free(bath_hamiltonian);
    free(interaction);
}
void wrapper_heat_driven(double *heat, double *state_r, double *state_i, double *bath_state_r, double *bath_state_i, double *bath_hamiltonian_r, double *bath_hamiltonian_i, double *interaction_r, double *interaction_i, int dim, double tf, double dt)
{
    int N = (int)(tf/dt), i;
    double complex *state, *bath_state, *bath_hamiltonian, *interaction;
    state = (double complex*) calloc(N*dim*dim, sizeof(double complex));
    bath_state = (double complex*) calloc(dim*dim, sizeof(double complex));
    bath_hamiltonian = (double complex*) calloc(dim*dim, sizeof(double complex));
    interaction = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    for(i = 0; i < N*dim*dim; i++)
    {
        *(state + i) = *(state_r + i) + *(state_i + i)*I;
    }
    for(i = 0; i < dim*dim; i++)
    {
        *(bath_state + i) = *(bath_state_r + i) + *(bath_state_i + i)*I;
        *(bath_hamiltonian + i) = *(bath_hamiltonian_r + i) + *(bath_hamiltonian_i + i)*I;
    }
    for(i = 0; i < dim*dim*dim*dim; i++)
    {
        *(interaction + i) = *(interaction_r + i) + *(interaction_i + i)*I;
    }
    calc_heat_driven(heat, state, bath_state, bath_hamiltonian, interaction, dim, tf, dt);    
    free(state);
    free(bath_state);
    free(bath_hamiltonian);
    free(interaction);
}