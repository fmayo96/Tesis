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
#include "energy.h"
int main()
{
double beta = 1, h = 0.75, z = 2*cosh(beta*h), epsilon = sqrt(5), dt = 0.001, tf = 15;
int i, j, dim = 2, N_steps = (int)(tf/dt);
double complex *state, *hamiltonian, *bath_state, *bath_hamiltonian, *interaction;
state = (double complex*) calloc(N_steps*dim*dim, sizeof(double complex));
hamiltonian = (double complex*) calloc(N_steps*dim*dim, sizeof(double complex));
bath_state = (double complex*) calloc(dim*dim, sizeof(double complex));
bath_hamiltonian = (double complex*) calloc(dim*dim, sizeof(double complex));
interaction = (double complex*) calloc(dim*dim*dim*dim, sizeof(double complex));
double *energy, *heat, *work;
energy = (double*) calloc(N_steps, sizeof(double));
heat = (double*) calloc(N_steps, sizeof(double));
work = (double*) calloc(N_steps, sizeof(double));
*(bath_state) = exp(-beta*h)/z;
*(bath_state + 1) = 0;
*(bath_state + 2) = 0;
*(bath_state + 3) = exp(beta*h)/z;
*(state) = exp(-beta*h)/z;
*(state + 1) = 0;
*(state + 2) = 0;
*(state + 3) = exp(beta*h)/z;
*(bath_hamiltonian) = h;
*(bath_hamiltonian + 1) = 0;
*(bath_hamiltonian + 2) = 0;
*(bath_hamiltonian + 3) = -h;
for(i = 0; i < N_steps; i++)
{
    *(hamiltonian + dim*dim*i) = h;
    *(hamiltonian + dim*dim*i + 1) = 0;
    *(hamiltonian + dim*dim*i + 2) = 0;
    *(hamiltonian + dim*dim*i + 3) = -h;

}
*(interaction) = 0;
*(interaction + 1) = 0;
*(interaction + 2) = 0;
*(interaction + 3) = epsilon;
*(interaction + 4) = 0;
*(interaction + 5) = 0;
*(interaction + 6) = 0;
*(interaction + 7) = 0;
*(interaction + 8) = 0;
*(interaction + 9) = 0;
*(interaction + 10) = 0;
*(interaction + 11) = 0;
*(interaction + 12) = epsilon;
*(interaction + 13) = 0;
*(interaction + 14) = 0;
*(interaction + 15) = 0;

Open_evolution(state, hamiltonian, bath_state, bath_hamiltonian, interaction, dim, tf, dt);
calc_energy(energy, state, hamiltonian, dim, tf, dt);
calc_heat(heat, state, bath_state, bath_hamiltonian, interaction, dim, tf, dt);
calc_work(work, energy, heat, N_steps);
char filename[255];
sprintf(filename,"test_modulos.txt");
FILE *fp=fopen(filename,"w");

for(i = 0; i < N_steps; i++)
{
    fprintf(fp, "%lf %lf %lf\n", *(energy + i),*(heat + i), *(work + i));
}
fclose(fp);
free(state);

free(hamiltonian);
free(bath_state);
free(bath_hamiltonian);
free(interaction);
//free(t);
return 0;
}
