#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double f_hot(double x, double y, double bh, double eps);
double g_hot(double x, double y, double bh, double eps);

int hot_isochore(double *X, double *Y, double = X0, double = Y0, double N_final, double N_initial, double dt, double bh, double eps)
{
int N = (int) (t_final / dt);
X = (double*) malloc(N * sizeof(double));
Y = (double*) malloc(N * sizeof(double));
double K_1, K_2, K_3, K_4, J_1, J_2, J_3, J_4;
int i;

*X = X0;
*Y = Y0;
for(i = 0; i < N; i++)
  {
    K_1 = f(*(X + i), *(Y + i));
    J_1 = g(*(X + i), *(Y + i));
    K_2 = f(*(X + i) + dt/2.0 * K_1, *(Y + i) + dt/2.0 * J_1);
    J_2 = g(*(X + i) + dt/2.0 * K_1, *(Y + i) + dt/2.0 * J_1);
    K_3 = f(*(X + i) + dt/2.0 * K_2, *(Y + i) + dt/2.0 * J_2);
    J_3 = g(*(X + i) + dt/2.0 * K_2, *(Y + i) + dt/2.0 * J_2);
    K_4 = f(*(X + i) + dt * K_3, *(Y + i) + dt * J_3);
    J_4 = g(*(X + i) + dt * K_3, *(Y + i) + dt * J_3);
    *(X + i + 1) = *(X + i) + dt/6.0 * (K_1 + 2 * K_2 + 2 * K_3 + K_4);
    *(Y + i + 1) = *(Y + i) + dt/6.0 * (J_1 + 2 * J_2 + 2 * J_3 + J_4);
  }
char filename[255];
sprintf(filename,"TestRK4.txt");
FILE *fp=fopen(filename,"w");

for(i = 0; i < N; i++)
    {
      fprintf(fp, "%lf %lf \n", *(X + i), *(Y + i));
    }
fclose(fp);

return 0;
}
double f(double x, double y, double bh, double eps)
{
double  alpha = eps * eps / (2 * cosh(bh));
double gamma_pl = exp(-bh) * alpha, gamma_min = exp(bh) * alpha;
double m =  gamma_pl * y - gamma_min * x;
return m;
}
double g_hot(double x, double y, double bh, double eps)
{
double  alpha = eps * eps / (2 * cosh(bh));
double gamma_pl = exp(-bh) * alpha, gamma_min = exp(bh) * alpha;
double m =  -gamma_pl * y + gamma_min * x;
return m;
}
