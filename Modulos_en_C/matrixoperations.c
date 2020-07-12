int dot(double complex *C, double complex *A, double complex *B, int dim)
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
    return 0;
}
int kron(double complex *C, double complex *A, double complex *B, int dim)
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

    return 0;
}
int trace(double complex *A, int dim)
{
    int i;
    double trace = 0;
    for(i = 0; i < dim; i++)
        {
            trace += *(A + i * dim + i);
        }
    return trace;
}
int partial_trace(double complex *m, double complex *M, int dim)//M = kron(A,B)
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

    return 0;
}
int commutator(double complex *C, double complex *A, double complex *B, int dim)
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
    return 0;
}
