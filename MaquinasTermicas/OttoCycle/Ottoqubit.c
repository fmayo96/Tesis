#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "hotisochore.h"
#include "coldisochore.h"

int main()
{
int N_cycles = 10000, i, j, N_power = 5;
double *Power;
Power = (double*) malloc(N_power * sizeof(double));
double W, Q_hot, Eff;
double eps = sqrt(5), beta_hot = 1, beta_cold = 5, h_hot = 2, h_cold = 1;
double bh_hot = beta_hot * h_hot / 2.0 , bh_cold =beta_cold * h_cold / 2.0, t_cycle, dt = 0.000001;
double /*Z_hot = exp(bh_hot) + exp(bh_hot),*/ Z_cold = exp(-bh_cold) + exp(bh_cold);
int N_cycle, N, N_cycle_max =(int)((double)N_power * 0.01 / dt);
double *X, *Y, *t;
N = N_cycles * N_cycle_max;
X = (double*) malloc(N * sizeof(double));
Y = (double*) malloc(N * sizeof(double));
t = (double*) malloc(N * sizeof(double));


for(j = 0; j< N_power; j++)
{
t_cycle = (j + 1) * 0.01;
N_cycle = (t_cycle/dt);
printf("%lf\n", t_cycle);
*X = exp(-bh_cold)/Z_cold;
*Y = exp(bh_cold)/Z_cold;
for(i = 0 ; i < N_cycles; i++)
  {
    hot_isochore(X, Y, (int)((2*i + 1) * N_cycle / 2),(int)((2*i) * N_cycle / 2), dt, bh_hot, eps);
    cold_isochore(X, Y, (int)((2*i + 2) * N_cycle / 2),(int)((2*i + 1) * N_cycle / 2), dt, bh_cold, eps);
  }



W = (h_hot - h_cold)/2 * (*(X + (N_cycles - 1) * N_cycle + (int)(N_cycle/2)) - *(Y + (N_cycles - 1) * N_cycle + (int)(N_cycle/2))) + (h_cold - h_hot)/2 * (*(X + N_cycles * N_cycle) - *(Y + N_cycles * N_cycle));
Q_hot = h_hot/2 * (*(X + (N_cycles - 1) * N_cycle + (int)(N_cycle/2)) - *(Y + (N_cycles - 1) * N_cycle + (int)(N_cycle/2))) - h_hot/2 * (*(X + (N_cycles - 1) * N_cycle) - *(Y + (N_cycles - 1) * N_cycle));
Eff = W/Q_hot;
*(Power + j) = W / t_cycle;
printf("Work = %lf \n", W);
printf("Heat = %lf \n", Q_hot);
printf("Efficiency = %lf \n", Eff);
/*
char filename[255];
sprintf(filename,"test_otto.txt");
FILE *fp=fopen(filename,"w");
*t = 0;
for(i = 0; i < N; i++)
  {
    *(t + i + 1) = *(t + i) + dt;
  }

for(i = 0; i < N; i++)
  {
    fprintf(fp, "%lf %lf %lf \n", *(t + i), *(X + i), *(Y + i));
  }
*/
}

char filename[255];
sprintf(filename,"Power_0.01-0.05.txt");
FILE *fp=fopen(filename,"w");

for(j = 0; j< N_power; j++)
  {
    fprintf(fp, "%lf %lf \n", ((double)(j+1)*0.01), *(Power + j));
  }

free(Power);
free(X);
free(Y);
free(t);
fclose(fp);

return 0;
}

#include "hotisochore.c"
#include "coldisochore.c"
