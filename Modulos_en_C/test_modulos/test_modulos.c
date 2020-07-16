#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include "matrixoperations.h"
#include "propagators.h"
#include "RK4.h"
#include "time_evolution.h"
#include "wrappers.h"
int main()
{
double beta = 1, h = 0.75, z = 2*cosh(beta*h), epsilon = sqrt(5), dt = 0.001, tf = 15;
int i, j, dim = 2, N_steps = (int)(tf/dt);
double *state_r, *state_i, *hamiltonian_r, *hamiltonian_i, *bath_state_r, *bath_state_i, *bath_hamiltonian_r, *bath_hamiltonian_i, *interaction_r, *interaction_i;
//double *t;
//t = (double*) calloc(N_steps, sizeof(double));
/*state = (double complex*) calloc(N_steps*dim*dim, sizeof(double complex));
hamiltonian = (double complex*) calloc(N_steps*dim*dim, sizeof(double complex));
bath_state = (double complex*) calloc(dim*dim, sizeof(double complex));
bath_hamiltonian = (double complex*) calloc(dim*dim, sizeof(double complex));
interaction = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
*/
state_r = (double*) calloc(N_steps*dim*dim, sizeof(double));
state_i = (double*) calloc(N_steps*dim*dim, sizeof(double));
hamiltonian_r = (double*) calloc(N_steps*dim*dim, sizeof(double));
hamiltonian_i = (double*) calloc(N_steps*dim*dim, sizeof(double));
bath_state_r = (double*) calloc(dim*dim, sizeof(double));
bath_state_i = (double*) calloc(dim*dim, sizeof(double));
bath_hamiltonian_r = (double*) calloc(dim*dim, sizeof(double));
bath_hamiltonian_i = (double*) calloc(dim*dim, sizeof(double));
interaction_r = (double*) calloc(dim*dim*dim*dim, sizeof(double));
interaction_i = (double*) calloc(dim*dim*dim*dim, sizeof(double));


*(bath_state_r) = exp(-beta*h)/z;
*(bath_state_r + 1) = 0;
*(bath_state_r + 2) = 0;
*(bath_state_r + 3) = exp(beta*h)/z;
*(state_r) = exp(-beta*h)/z;
*(state_r + 1) = 0;
*(state_r + 2) = 0;
*(state_r + 3) = exp(beta*h)/z;
*(bath_hamiltonian_r) = h;
*(bath_hamiltonian_r + 1) = 0;
*(bath_hamiltonian_r + 2) = 0;
*(bath_hamiltonian_r + 3) = -h;
for(i = 0; i < N_steps; i++)
{
    *(hamiltonian_r + dim*dim*i) = h;
    *(hamiltonian_r + dim*dim*i + 1) = 0;
    *(hamiltonian_r + dim*dim*i + 2) = 0;
    *(hamiltonian_r + dim*dim*i + 3) = h;

}
*(interaction_r) = 0;
*(interaction_r + 1) = 0;
*(interaction_r + 2) = 0;
*(interaction_r + 3) = epsilon;
*(interaction_r + 4) = 0;
*(interaction_r + 5) = 0;
*(interaction_r + 6) = 0;
*(interaction_r + 7) = 0;
*(interaction_r + 8) = 0;
*(interaction_r + 9) = 0;
*(interaction_r + 10) = 0;
*(interaction_r + 11) = 0;
*(interaction_r + 12) = epsilon;
*(interaction_r + 13) = 0;
*(interaction_r + 14) = 0;
*(interaction_r + 15) = 0;

wrapper_Open_evolution(state_r, state_i, hamiltonian_r, hamiltonian_i, bath_state_r, bath_state_i, bath_hamiltonian_r, bath_hamiltonian_i, interaction_r, interaction_i, dim, tf, dt);

char filename[255];
sprintf(filename,"test_modulos.txt");
FILE *fp=fopen(filename,"w");

for(i = 0; i < N_steps; i++)
{
    for(j = 0; j < dim*dim; j++)
    {
        fprintf(fp, "%lf ", creal(*(state_r + i*dim*dim + j)));
    }
fprintf(fp, "\n");
}
fclose(fp);
free(state_r);
free(state_i);
free(hamiltonian_r);
free(hamiltonian_i);
free(bath_state_r);
free(bath_state_i);
free(bath_hamiltonian_r);
free(bath_hamiltonian_i);
free(interaction_r);
free(interaction_i);
//free(t);
return 0;
}
