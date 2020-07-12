#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "matrixoperations.h"
#include "propagators.h"
#include "RK4.h"

int Time_evolution(double complex *state, double complex *hamiltonian, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim, double tf, double dt)
{
    int N = (int)(tf/dt), step;
    double complex *propagator, *dissipator;
    propagator = (double complex*) calloc(dim*dim, sizeof(double complex));
    dissipator = (double complex*) calloc(dim*dim, sizeof(double complex));
    for(paso = 0; step < N; step++)
    {
        RK4(propagator, dissipator, state, hamiltonian, bath_state, bath_hamiltonian, interaction, dt, dim, step);
    }
    free(propagator);
    free(dissipator);
    return 0;
}
/*int Driven_evolution(double complex *state, double complex *hamiltonian, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim, double tf, double dt)
{
    int N = (int)(tf/dt);
    double complex *propagator, *dissipator;
    propagator = (double complex*) calloc(dim*dim, sizeof(double complex));
    dissipator = (double complex*) calloc(dim*dim, sizeof(double complex));
    for(i = 0; i < N; i++)
    {
        Dissipator(dissipator, state, bath_state, interaction, dim);
        Propagator(propagator, state, hamiltonian, dissipator, dim);
        RK4(propagator, state, dt, dim);
        Unitary(propagator, state, hamiltonian, dim);
        RK4(propagator, state, dt, dim);
    }

    return 0;
}
*/
#include "matrixoperations.c"
#include "propagators.c"
#include "RK4.c"
