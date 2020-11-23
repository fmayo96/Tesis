#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "matrixoperations.h"

void calc_energy(double *energy, double complex *state, double complex *hamiltonian, int dim, double tf, double dt)
{
    int i, j, N = (int)(tf/dt);
    double complex *C, *step_state, *step_hamiltonian;
    C = (double complex*) calloc(dim*dim, sizeof(double complex));
    step_state = (double complex*) calloc(dim*dim, sizeof(double complex));
    step_hamiltonian = (double complex*) calloc(dim*dim, sizeof(double complex));
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < dim*dim; j++)
        {
            *(step_state + j) = *(state + dim*dim*i + j);
            *(step_hamiltonian + j) = *(hamiltonian + dim*dim*i + j);
        }
        dot(C, step_state, step_hamiltonian, dim);
        *(energy + i) = creal(trace(C, dim));    
    }
    
    free(step_state);
    free(step_hamiltonian);
    free(C);
}
void calc_heat(double *heat, double complex *state, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim,double tf, double dt)
{
    int i, j, N = (int)(tf/dt);
    double complex *step_state, *product_state, *product_hamiltonian, *eye, *comm, *double_comm, *product_double_comm_state;
    step_state = (double complex*) calloc(dim*dim, sizeof(double complex));
    product_state = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    product_hamiltonian = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    comm = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    double_comm = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    product_double_comm_state = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    eye = (double complex*) calloc(dim*dim, sizeof(double complex));
    for(i = 0; i < dim*dim; i++)
    {
        *(eye + dim*i + i) = 1;
    }
    kron(product_hamiltonian, eye, bath_hamiltonian, dim);   
    for(i = 0; i < N-1; i++)
    {
        for(j = 0; j < dim*dim; j++)
        {
            *(step_state + j) = *(state + dim*dim*i + j);
        }
        kron(product_state, step_state, bath_state, dim);
        commutator(comm, interaction, product_hamiltonian, dim*dim);
        commutator(double_comm, interaction, comm, dim*dim);
        dot(product_double_comm_state, double_comm, product_state, dim*dim);
        *(heat + i + 1) = *(heat + i) + 0.5 * dt * trace(product_double_comm_state, dim*dim);
    }
    free(step_state);
    free(product_state);
    free(product_hamiltonian);
    free(eye);
    free(comm);
    free(double_comm);
    free(product_double_comm_state);
}
void calc_heat_driven(double *heat, double complex *state, double complex *bath_state, double complex *bath_hamiltonian, double complex *interaction, int dim,double tf, double dt)
{
    int i, j, N = (int)(tf/dt);
    double complex *step_state, *product_state, *product_hamiltonian, *eye, *comm, *double_comm, *product_double_comm_state;
    step_state = (double complex*) calloc(dim*dim, sizeof(double complex));
    product_state = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    product_hamiltonian = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    comm = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    double_comm = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    product_double_comm_state = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
    eye = (double complex*) calloc(dim*dim, sizeof(double complex));
    for(i = 0; i < dim*dim; i++)
    {
        *(eye + dim*i + i) = 1;
    }
    kron(product_hamiltonian, eye, bath_hamiltonian, dim);   
    for(i = 0; i < N-1; i += 2)
    {
        for(j = 0; j < dim*dim; j++)
        {
            *(step_state + j) = *(state + dim*dim*i + j);
        }
        kron(product_state, step_state, bath_state, dim);
        commutator(comm, interaction, product_hamiltonian, dim*dim);
        commutator(double_comm, interaction, comm, dim*dim);
        dot(product_double_comm_state, double_comm, product_state, dim*dim);
        *(heat + i + 1) = *(heat + i) + 0.5 * dt * trace(product_double_comm_state, dim*dim);
        *(heat + i + 2) = *(heat + i + 1);
    }
    free(step_state);
    free(product_state);
    free(product_hamiltonian);
    free(eye);
    free(comm);
    free(double_comm);
    free(product_double_comm_state);
}
void calc_work(double *work, double *energy, double *heat, int N)
{
    int i;
    for(i = 0; i < N; i++)
    {
        *(work + i) = *(heat + i) - *(energy + i);
    }
}