#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrixoperations.h"
int dot(double *C, double *A, double *B, int dim);
int kron(double *C, double *A, double *B, int dim);
int trace(double *A, int dim);
int partial_trace(double *m, double *M, int dim);
int main()
{
double *A, *B, *C, *m;
int rows_A = 2, cols_A = 2, rows_B = 2, cols_B = 2, i, j;
A = (double*) malloc(rows_A * cols_A * sizeof(double));
B = (double*) malloc(rows_B * cols_B * sizeof(double));
C = (double*) calloc(rows_A * cols_A * rows_B * cols_B , sizeof(double)); 
m = (double*) calloc(rows_A * cols_A, sizeof(double));
*(A + 0 * rows_A + 0) = 1;
*(A + 0 * rows_A + 1) = 0;
*(A + 1 * rows_A + 0) = 0;
*(A + 1 * rows_A + 1) = -1;

*(B + 0 * rows_B + 0) = 0;
*(B + 0 * rows_B + 1) = 1;
*(B + 1 * rows_B + 0) = 1;
*(B + 1 * rows_B + 1) = 0;

commutator(C,A,B,cols_A);
for(i = 0; i < cols_A; i++)
{
    for(j = 0; j < cols_A; j++)
    {
        printf("%lf ", *(C + i*cols_A + j));
    }
    printf("\n");
}

/*
for(i = 0; i < rows_A * rows_B; i++)
{
    for(j = 0; j < cols_A * cols_B; j++)
    {
        printf("%lf ", *(C + rows_A * rows_B * i + j));
    }
    printf("\n");
}
printf("\n");
kron(C, A, B, cols_A);

for(i = 0; i < rows_A * rows_B; i++)
{
    for(j = 0; j < cols_A * cols_B; j++)
    {
        printf("%lf ", *(C + cols_A * cols_B * i + j));
    }
    printf("\n");
}
partial_trace(m, C, cols_A);
for(i = 0; i < cols_A; i++)
{
    for(j = 0; j < cols_A; j++)
    {
        printf("%.0lf ", *(m + i*cols_A + j));
    }
printf("\n");

}*/
return 0;
}
#include "matrixoperations.c"