#ifndef MATRIXOPERATIONS_H
#define MATRIXOPERATIONS_H
int dot(double complex *C, double complex *A, double complex *B, int dim);
int kron(double complex *C, double complex *A, double complex *B, int dim);
int trace(double complex *A, int dim);
int partial_trace(double complex *m, double complex *M, int dim);
int commutator(double complex *C, double complex *A, double complex *B, int dim);
#endif
