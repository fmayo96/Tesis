#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double Fidelity(double x, double rho_tf, double *h);
int main()
{
    //--------------------------DEFINO VARIABLES----------------------------------------
    double dt = 0.01, tf = 10, h_init = 1, rho_tf, omega2 = (double)h_init*h_init/4, dh;
    int i, j, N_time = (int)tf/dt, N_param = 100;
    double beta = 1, Z = 2*cosh((double)beta*h/2); 
    double *h, *rho, *t, *fidelity;
    h = (double*) calloc(N_time, double);
    rho = (double*) calloc(N_time, double);
    fidelity = (double*) calloc(N_time, double);
    *rho = exp(-beta*h_init)/Z;
    *(rho + N_time) = rho_tf;
    for(i = 0; i < N_time; i++)
    {
        *(h + i) = h_init;
    }
    rho_tf = exp(beta*h_init)/Z;
    //-------------------------------------------------------------------------------------



    return 0;
}
double Fidelity(double x, double rho_tf, double *h)
{
    double fidelity = rho_tf - x;
    if (fidelity < 0) {fidelity = -fidelity};
    return fidelity;
}
double Gradient_descent(double h, double rho, double rho_tf, int N_param, double h_init, double dh)
{
    int i;
    double step, dh = 0.001, grad_rho;

    for(i = 0; i < N_param; i++)
    {
        grad_rho = (double)(Rho(h + dh, t) - Rho(h, t))/dh;
        h -= step * 2 * (rho_tf - rho) * (-grad_rho); 
    }
    return h;
}
double Rho(double h, double t)
{
    double h_init = 1;
    double epsilon = sqrt(0.5);
    double beta = 1, Z = 2*cosh((double)beta*h_init/2); 
    double rho = (exp(-(double)beta*h_init/2) - exp((double)beta*h_init/2)) * exp(-(double)epsilon*epsilon*t/2) * (h*h + h*sqrt(h_init*h_init-h*h)*cos(2*h_init*t))
    + (16*h*h*exp((double)beta*h_init/2)/Z + epsilon*epsilon*epsilon*epsilon*exp((double)beta*h_init/2)/Z - 8*(h_init*h_init-h*h)) / (16*h_init*h_init + epsilon*epsilon);
    return rho;
}