#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "matrixoperations.h"
void Dissipator(double complex *dissipator, double complex *state, double complex *bath_state, double complex *interaction, int dim)
{   
    int i;
    double complex *rho, *comm_v_rho, *double_comm;
    rho = (double complex*)calloc(dim*dim*dim*dim, sizeof(double complex));
    comm_v_rho = (double complex*)calloc(dim*dim*dim*dim, sizeof(double complex));
    double_comm = (double complex*)calloc(dim*dim*dim*dim, sizeof(double complex));
    kron(rho, state, bath_state, dim);
    commutator(comm_v_rho, interaction, rho, dim*dim);
    commutator(double_comm, interaction, comm_v_rho, dim*dim);
    partial_trace(dissipator, double_comm, dim);
    for(i = 0; i < dim*dim; i++)
    {
        *(dissipator + i) = -*(dissipator + i) * 0.5;
    }
}
void Propagator(double complex *propagator, double complex *state, double complex *hamiltonian, double complex *dissipator, int dim)
{
    int i;
    commutator(propagator, hamiltonian, state, dim);
    for(i = 0; i < dim*dim; i++)
    {
        *(propagator + i) = *(propagator + i) *(-1*I);
        *(propagator + i) += *(dissipator + i);
    }
}
void Unitary(double complex *propagator, double complex *state, double complex*hamiltonian, int dim)
{
    int i;
    commutator(propagator, hamiltonian, state, dim);
    for(i = 0; i < dim*dim; i++)
    {
        *(propagator + i) = *(propagator + i) *(-1*I);
    }

}

