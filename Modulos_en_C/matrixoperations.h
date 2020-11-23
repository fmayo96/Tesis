#ifndef MATRIXOPERATIONS_H
#define MATRIXOPERATIONS_H
void dot(double complex *C, double complex *A, double complex *B, int dim);
void kron(double complex *C, double complex *A, double complex *B, int dim);
double trace(double complex *A, int dim);
void partial_trace(double complex *m, double complex *M, int dim);
void commutator(double complex *C, double complex *A, double complex *B, int dim);
#endif
