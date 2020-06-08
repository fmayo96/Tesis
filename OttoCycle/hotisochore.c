#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double f_hot(double x, double y, double bh, double eps);
double g_hot(double x, double y, double bh, double eps);

int hot_isochore(double *X, double *Y, int N_final, int N_initial, double dt, double bh, double eps)
{
double K_1, K_2, K_3, K_4, J_1, J_2, J_3, J_4;
int i;

for(i = N_initial; i < N_final; i++)
  {
    K_1 = f_hot(*(X + i), *(Y + i), bh, eps);
    J_1 = g_hot(*(X + i), *(Y + i), bh, eps);
    K_2 = f_hot(*(X + i) + dt/2.0 * K_1, *(Y + i) + dt/2.0 * J_1, bh, eps);
    J_2 = g_hot(*(X + i) + dt/2.0 * K_1, *(Y + i) + dt/2.0 * J_1, bh, eps);
    K_3 = f_hot(*(X + i) + dt/2.0 * K_2, *(Y + i) + dt/2.0 * J_2, bh, eps);
    J_3 = g_hot(*(X + i) + dt/2.0 * K_2, *(Y + i) + dt/2.0 * J_2, bh, eps);
    K_4 = f_hot(*(X + i) + dt * K_3, *(Y + i) + dt * J_3, bh, eps);
    J_4 = g_hot(*(X + i) + dt * K_3, *(Y + i) + dt * J_3, bh, eps);
    *(X + i + 1) = *(X + i) + dt/6.0 * (K_1 + 2 * K_2 + 2 * K_3 + K_4);
    *(Y + i + 1) = *(Y + i) + dt/6.0 * (J_1 + 2 * J_2 + 2 * J_3 + J_4);
  }
return 0;
}
double f_hot(double x, double y, double bh, double eps)
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
