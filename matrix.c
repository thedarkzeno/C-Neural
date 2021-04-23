#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "matrix.h"
#include "activation.h"

Matrix createMatrix(int lines, int columns)
{
    Matrix new = malloc(sizeof(struct matrix));
    if (!new)
        return NULL;
    new->lines = lines;
    new->columns = columns;
    new->values = (double **)malloc(sizeof(double *) * lines);
    int x = 0;
    for (x = 0; x < lines; x++)
    {
        new->values[x] = (double *)calloc(columns, sizeof(double));
    }
    return new;
}

void freeMatrix(Matrix M){
    if(M!=NULL){
    int i = 0;
    for(i=0;i<M->lines;i++){
        free(M->values[i]);
    }
    free(M);}
}

void initMatrix(Matrix M, double *values)
{
    int lines = M->lines;
    int columns = M->columns;

    int i, j;
    for (i = 0; i < lines; i++)
    {
        for (j = 0; j < columns; j++)
        {
            M->values[i][j] = values[i * columns + j];
        }
    }
}

void initMatrixRandom(Matrix M)
{
    int lines = M->lines;
    int columns = M->columns;

    int i, j;
    for (i = 0; i < lines; i++)
    {
        for (j = 0; j < columns; j++)
        {
            M->values[i][j] = 2*(double)rand() / (double)RAND_MAX - 1;
        }
    }
}

double getMatrixElement(Matrix M, int line, int column)
{
    return M->values[line][column];
}

void matrixMultiplication(Matrix m1, Matrix m2, Matrix result)
{
    // Matrix result = createMatrix(m1->lines, m2->columns);
    int i, j, k;

    for (i = 0; i < m1->lines; i++)
    {
        for (j = 0; j < m2->columns; j++)
        {
            result->values[i][j] = 0;
            for (k = 0; k < m1->columns; k++)
            {
                // printf("A[%d][%d] += a%d%d * b%d%d = %f*%f\n", i, j, i,k,k,j, m1->values[i][k], m2->values[k][j]);
                result->values[i][j] += m1->values[i][k] * m2->values[k][j];
            }
        }
    }
    // return result;
}

void matrixDotProduct(Matrix m1, Matrix m2, Matrix result)
{
    // Matrix result = createMatrix(m1->lines, m2->columns);
    int i, j;

    for (i = 0; i < m1->lines; i++)
    {
        for (j = 0; j < m2->columns; j++)
        {
            result->values[i][j] = m1->values[i][j] * m2->values[i][j];
        }
    }
    // return result;
}

void matrixSum(Matrix m1, Matrix m2, Matrix result)
{
    // Matrix result = createMatrix(m1->lines, m2->columns);
    int i, j;

    for (i = 0; i < m1->lines; i++)
    {
        for (j = 0; j < m2->columns; j++)
        {
            result->values[i][j] = m1->values[i][j] + m2->values[i][j];
        }
    }
    // return result;
}

void matrixSumBias(Matrix m1, Matrix m2, Matrix result)
{
    // Matrix result = createMatrix(m1->lines, m2->columns);
    int i, j;

    for (i = 0; i < m1->lines; i++)
    {
        for (j = 0; j < m2->columns; j++)
        {
            result->values[i][j] = m1->values[i][j] + m2->values[0][j];
        }
    }
    // return result;
}

void matrixSubtraction(Matrix m1, Matrix m2, Matrix result)
{
    // Matrix result = createMatrix(m1->lines, m2->columns);
    int i, j;

    for (i = 0; i < m1->lines; i++)
    {
        for (j = 0; j < m2->columns; j++)
        {
            result->values[i][j] = m1->values[i][j] - m2->values[i][j];
        }
    }
    // return result;
}

Matrix matrixClone(Matrix m)
{
    Matrix result = createMatrix(m->lines, m->columns);
    int i, j;

    for (i = 0; i < m->lines; i++)
    {
        for (j = 0; j < m->columns; j++)
        {
            result->values[i][j] = m->values[i][j];
        }
    }
    return result;
}

void matrixTranspose(Matrix m, Matrix result)
{
    // Matrix result = createMatrix(m->columns, m->lines);
    int i, j;

    for (i = 0; i < result->lines; i++)
    {
        for (j = 0; j < result->columns; j++)
        {
            result->values[i][j] = m->values[j][i];
        }
    }
    // return result;
}

void matrixMultiplyByEscalar(Matrix m, double x, Matrix result)
{
    // Matrix result = createMatrix(m->lines, m->columns);
    int i, j;

    for (i = 0; i < m->lines; i++)
    {
        for (j = 0; j < m->columns; j++)
        {
            result->values[i][j] = m->values[i][j] * x;
        }
    }
    // return result;
}

void activate(Matrix M, int prime, char activation[4], Matrix result)
{
    // Matrix result = createMatrix(M->lines, M->columns);
    int i, j;

    for (i = 0; i < M->lines; i++)
    {
        for (j = 0; j < M->columns; j++)
        {
            if (prime == 1)
            {
                if (strcmp(activation, "tanh") == 0)
                {
                    result->values[i][j] = Tanh(M->values[i][j], 1);
                }
                else if (strcmp(activation, "sigm") == 0)
                {
                    result->values[i][j] = Sigmoid(M->values[i][j], 1);
                }
            }
            else
            {
                if (strcmp(activation, "tanh") == 0)
                {
                    result->values[i][j] = Tanh(M->values[i][j], 0);
                }
                else if (strcmp(activation, "sigm") == 0)
                {
                    result->values[i][j] = Sigmoid(M->values[i][j], 0);
                }
            }
        }
    }
    // return result;
}

void printMatrix(Matrix M)
{

    int i, j;

    for (i = 0; i < M->lines; i++)
    {
        for (j = 0; j < M->columns; j++)
        {
            printf("%f ", M->values[i][j]);
        }
        printf("\n");
    }
}

double MatrixSumValues(Matrix M)
{
    int i, j;
    double result;

    for (i = 0; i < M->lines; i++)
    {
        for (j = 0; j < M->columns; j++)
        {
            result += M->values[i][j];
        }
    }
    return result;
}

int matrixCompareMax(Matrix m1, Matrix m2)
{

    int c = m1->columns;
    int l = m1->lines;
    int i = 0, j = 0, count = 0;
    for (i = 0; i < l; i++)
    {
        int k1, k2;
        double v1, v2;
        for (j = 0; j < c; j++)
        {
            if (j == 0)
            {
                k1 = j;
                v1 = m1->values[i][j];
                k2 = j;
                v2 = m2->values[i][j];
            }
            else
            {
                if (m1->values[i][j] > v1)
                {
                    k1 = j;
                    v1 = m1->values[i][j];
                }
                if (m2->values[i][j] > v2)
                {
                    k2 = j;
                    v2 = m2->values[i][j];
                }
            }
        }
        if (k1 == k2)
        {
            count++;
        }
        // printf("k1 %d k2 %d\n", k1, k2);
    }

    return count;
}


// double nan_to_num(double num){
//   if (isinf(num)){
//     if (signbit(num))
//       return -MAXdouble;
//     else
//       return MAXdouble;
//   } else {
//     return num;
//   }
// }
