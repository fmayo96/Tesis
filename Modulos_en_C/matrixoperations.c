#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
void dot(double complex *C, double complex *A, double complex *B, int dim)
{
    int i, j, k;
    for(i = 0; i < dim*dim; i++)
    {
        *(C + i) = 0;
    }
    for(i = 0; i < dim; i++)
    {
        for(j = 0; j < dim; j++)
        {
            for(k = 0; k < dim; k++)
            {
                *(C + i*dim + j) += *(A + i*dim + k) * *(B + k*dim + j);
            }
        }
    }
}
void kron(double complex *C, double complex *A, double complex *B, int dim)
{
    int i, j, k, l;
    for(i = 0; i < dim; i++)
    {
		for(k = 0; k < dim; k++)
        {
			for(j = 0; j < dim; j++)
            {
				for(l = 0; l < dim; l++)
                {
					*(C + i * dim *dim *dim + k * dim * dim + j * dim + l) = *(A + i * dim + j) * *(B + k * dim + l);
				}
			}
		}
	}
}
double trace(double complex *A, int dim)
{
    int i;
    double trace = 0;
    for(i = 0; i < dim; i++)
        {
            trace += *(A + i * dim + i);
        }
    return creal(trace);
}
void partial_trace(double complex *m, double complex *M, int dim)//M = kron(A,B)
{
    int i, j, k;
    for(i = 0; i < dim*dim; i++)
    {
        *(m + i) = 0;
    }
    for(i = 0; i < dim; i++)
    {
        for(j = 0; j < dim; j++)
        {
            for(k = 0; k < dim; k++)
            {
                *(m + i*dim + j) += *(M + dim*dim*dim*i + j*dim + k*dim*dim + k);
            }         
        }
    } 
}
void commutator(double complex *C, double complex *A, double complex *B, int dim)
{
    int i;
    double complex *D, *E;
    D = (double complex*) calloc(dim*dim, sizeof(double complex));
    E = (double complex*) calloc(dim*dim, sizeof(double complex));
    dot(D,A,B,dim);
    dot(E,B,A,dim);
    for(i = 0; i < dim*dim; i++)
    {
        *(C + i) = *(D + i) - *(E + i);
    }
free(D);
free(E);
}
