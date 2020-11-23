#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <complex.h>

#define  eps sqrt(5)
#define beta_hot 0.2
#define h_hot 6.0
#define beta_cold 1.0
#define Z_hot (2 * cosh(beta_hot * (double)h_hot/2))


double Rho_00(double h, double omega, double t);
double Rho_01(double h, double omega,double t);
double Energy(double rho_00, double rho_01, double h, double omega, double t);
double Gradient_rho_00(double rho_00, double h, double t, double omega);
double Gradient_rho_01(double rho_01, double h, double t, double omega);
double Gradient_E(double rho_00, double rho_01, double grad_rho_00, double grad_rho_01, double h, double omega, double t);
double Gradient_Descent(double h, double omega, double t, double h_init);

int main()
{
    double tf = 10, dt = 0.001;
    int N_time = (int)(tf/dt), i;
    double *rho_00, *rho_01, *h, *t, *energy, h_init = 1, omega = (double) h_init/2;
    rho_00 = (double*) calloc(N_time, sizeof(double));
    rho_01 = (double*) calloc(N_time, sizeof(double));
    h = (double*) calloc(N_time, sizeof(double)); 
    t = (double*) calloc(N_time, sizeof(double)); 
    energy = (double*) calloc(N_time, sizeof(double)); 
    for(i = 0; i < N_time; i++)
    {
        *(h + i) = h_init * 0.5;
        *(t + i) = i * dt;
    }
    *h = h_init;
    *(h + N_time - 1) = h_init;
    for(i = 1; i < N_time-1; i++)
    {
        *(h + i) = Gradient_Descent(*(h + i), omega, *(t + i), h_init);
        printf("%lf\n", *(h + i));
    }
    for(i = 0; i < N_time; i++)
    {
        *(rho_00 + i) = Rho_00(*(h + i), omega, *(t + i));
        *(rho_01 + i) = Rho_01(*(h + i), omega, *(t + i));
        *(energy + i) = Energy(*(rho_00 + i), *(rho_01 + i), *(h + i), omega, *(t + i));
    }
    char filename[255];
    sprintf(filename,"test_rho.txt");
		
    FILE *fp=fopen(filename,"w");

    for(i = 0; i < N_time; i++)
    {
        fprintf(fp, "%.5lf %.5lf %.5lf\n", *(rho_00 + i), *(rho_01 + i), *(energy + i));
    }
    //printf("%lf \n", exp(-beta_hot*h_hot/2.0)/Z_hot);
    fclose(fp);
    free(rho_00);
    free(rho_01);
    free(h);
    free(t);
    return 0;
}
double Rho_00(double h, double omega, double t)
{
    double rho_00, Z_cold = 2 * cosh(beta_cold * (double)h / 2);
    rho_00   = (exp(-(double)beta_hot*h_hot/2.0)/Z_hot - exp((double)beta_cold*h/2.0)/Z_cold)/(omega*omega) * exp(-eps*eps*(double)t/2.0) * 
    (h*h/4.0 + h/2.0 * sqrt(omega*omega - (double)h*h/4.0) * cos(2*omega*t)) + (4*h*h * exp((double)beta_cold*h/2)/Z_cold + eps*eps*eps*eps*exp((double)beta_cold*h/2)/Z_cold
    - 8*(omega*omega - (double)h*h/4.0))/(16*omega*omega + eps*eps*eps*eps);
    return rho_00;
}
double Rho_01(double h, double omega,double t)
{
    double complex rho_01;
    double Z_cold = 2 * cosh(beta_cold * (double)h / 2);
    rho_01 = (0.5 * sqrt(omega*omega - h*h/4.0)/(omega*omega))*(exp(-(double)beta_hot*h_hot/2)/Z_hot - exp((double)beta_cold*h/2)/Z_cold) * exp(-eps*eps*(double)t/2) *
    (h*sin(omega*t)*sin(omega*t) + I*omega*sin(2*omega*t)) + 2*I*sqrt(omega*omega - h*h/4.0) * (1 + 2 * ((16*h*h/4.0 * exp((double)beta_cold*h/2)/Z_cold + eps*eps*eps*eps*exp((double)beta_cold*h/2)/Z_cold
    - 8*(omega*omega - h*h/4.0))/(16*omega*omega + eps*eps*eps*eps))) / (eps*eps + 2*I*h);
    return creal(rho_01);
}
double Energy(double rho_00, double rho_01, double h, double omega, double t)
{
    double energy; 
    energy = h * (rho_00 - 0.5) + 2 * sqrt(omega*omega - h*h/4.0) * rho_01;
    return energy;
}
double Gradient_rho_00(double rho_00, double h, double t, double omega)
{
    /*double Z_cold = 2 * cosh(beta_cold * (double)h / 2);
    double grad_rho_00 = (exp(-(double)beta_hot*h_hot/2.0)/Z_hot - exp((double)beta_cold*h/2.0)/Z_cold)/(omega*omega) * exp(-eps*eps*(double)t/2.0) * 
    (h/2.0 + 0.5 * sqrt(omega*omega - h*h/4.0)*cos(2*omega*t) - (double)h*h / (8.0*sqrt(omega*omega - h*h/4.0))) + 
    (8*h*exp((double)beta_cold*h/2)/Z_cold + 4.0*h) / (16*omega*omega + eps*eps*eps*eps); */
    double dh = 0.001;
    double grad_rho_00 = (double)(Rho_00(h + dh, omega, t) - Rho_00(h, omega, t))/ dh;
    return grad_rho_00;
}
double Gradient_rho_01(double rho_01, double h, double t, double omega)
{
    /*double Z_cold = 2 * cosh(beta_cold * (double)h / 2);
    double x = (4*h*h * exp((double)beta_cold*h/2)/Z_cold + eps*eps*eps*eps*exp((double)beta_cold*h/2)/Z_cold
    - 8*(omega*omega - (double)h*h/4.0))/(16*omega*omega + eps*eps*eps*eps);
    double complex grad_rho_01 = (exp(-(double)beta_hot*h_hot/2.0)/Z_hot - exp((double)beta_cold*h/2.0)/Z_cold)/(2.0*omega*omega) * exp(-eps*eps*(double)t/2.0) *
    (-h/(4.0 * sqrt(omega*omega - h*h /4.0)) * (h*sin(omega*t)*sin(omega*t) + I*omega*sin(2*omega*t)) + sqrt(omega*omega - h*h/4.0)*sin(omega*t)) 
    + 2*I*(-h/4.0)*(1 + 2*x)/(eps*eps + 2.0*I*h) + 2*I*sqrt(omega*omega - h*h/4.0) * 2 * ((8*h*exp((double)beta_cold*h/2)/Z_cold + 4.0*h) / (16*omega*omega + eps*eps*eps*eps) / (eps*eps + 2.0*I*h))
    + 2*I*sqrt(omega*omega - h*h/4.0) * (1 + 2*x) *(-1)*2*I/((eps*eps + 2*I*h)*(eps*eps + 2*I*h));*/
    double dh = 0.001;
    double grad_rho_01 = (double)(Rho_01(h+dh, omega, t) - Rho_01(h, omega, t))/ dh;
    return grad_rho_01;
}
double Gradient_E(double rho_00, double rho_01, double grad_rho_00, double grad_rho_01, double h, double omega, double t)
{
    double dh = 0.001;
    /*double grad_E = (rho_00 - 0.5) + h*grad_rho_00 - 2 * rho_01 * (h/2.0)/sqrt(omega*omega - h*h/4.0) 
    + 2*sqrt(omega*omega - h*h/4.0) * grad_rho_01;*/
    double grad_E = (double) (Energy(Rho_00(h + dh, omega, t), Rho_01(h + dh, omega, t), h + dh, omega, t) - Energy(Rho_00(h, omega, t), Rho_01(h, omega, t), h, omega, t)) / dh;
    return grad_E;
}
double Gradient_Descent(double h, double omega, double t, double h_init)
{
    double rho_00, rho_01, grad_rho_00, grad_rho_01, grad_E, alpha = 0.00001;
     for(int j = 0; j < 100000; j++)
    {
        if(h >= h_init) {h = h_init;}
        if(h <= -h_init) {h = -h_init;}
        rho_00 = Rho_00(h, omega, t);
        rho_01 = Rho_01(h, omega, t);
        grad_rho_00 = Gradient_rho_00(rho_00, h, t, omega);
        grad_rho_01 = Gradient_rho_01(rho_01, h, t, omega);
        grad_E = Gradient_E(rho_00, rho_01, grad_rho_00, grad_rho_01, h, omega, t);
        h += alpha * grad_E;
    }
    if(h >= h_init) {h = h_init;}
    if(h <= -h_init) {h = -h_init;}
    return h;
}