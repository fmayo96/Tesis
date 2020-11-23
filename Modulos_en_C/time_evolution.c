#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "matrixoperations.h"
#include "RK4.h"
#include "propagators.h"
#include "energy.h"
void Open_evolution(double complex *state, double complex *hamiltonian, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim, double tf, double dt)
{
    int N = (int)(tf/dt), step;
    double complex *propagator, *dissipator;
    propagator = (double complex*) calloc(dim*dim, sizeof(double complex));
    dissipator = (double complex*) calloc(dim*dim, sizeof(double complex));
    for(step = 0; step < N; step++)
    {
        RK4_open(propagator, dissipator, state, hamiltonian, bath_state, bath_hamiltonian, interaction, dt, dim, step);
    }
    free(propagator);
    free(dissipator);
}
void Driven_evolution(double complex *state, double complex *hamiltonian, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim, double tf, double dt)
{
    int N = (int)(tf/dt), step;
    double complex *propagator, *dissipator;
    propagator = (double complex*) calloc(dim*dim, sizeof(double complex));
    dissipator = (double complex*) calloc(dim*dim, sizeof(double complex));
    for(step = 0; step < N; step+=2)
    {
        RK4_open(propagator, dissipator, state, hamiltonian, bath_state, bath_hamiltonian, interaction, dt, dim, step);
        RK4_closed(propagator, state, hamiltonian, dt, dim, step + 1);
    }
    free(propagator);
    free(dissipator);
}
void Closed_evolution(double complex *state, double complex *hamiltonian, int dim, double tf, double dt)
{
    int N = (int)(tf/dt), step;
    double complex *propagator;
    propagator = (double complex*) calloc(dim*dim, sizeof(double complex));
    for(step = 0; step < N; step++)
    {
        RK4_closed(propagator, state, hamiltonian, dt, dim, step);
    }
    free(propagator);
}
