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
            M->values[i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

double getMatrixElement(Matrix M, int line, int column)
{
    return M->values[line][column];
}

Matrix matrixMultiplication(Matrix m1, Matrix m2)
{
    Matrix result = createMatrix(m1->lines, m2->columns);
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
    return result;
}

Matrix matrixDotProduct(Matrix m1, Matrix m2)
{
    Matrix result = createMatrix(m1->lines, m2->columns);
    int i, j;

    for (i = 0; i < m1->lines; i++)
    {
        for (j = 0; j < m2->columns; j++)
        {
            result->values[i][j] = m1->values[i][j] * m2->values[i][j];
        }
    }
    return result;
}

Matrix matrixSum(Matrix m1, Matrix m2)
{
    Matrix result = createMatrix(m1->lines, m2->columns);
    int i, j;

    for (i = 0; i < m1->lines; i++)
    {
        for (j = 0; j < m2->columns; j++)
        {
            result->values[i][j] = m1->values[i][j] + m2->values[i][j];
        }
    }
    return result;
}

Matrix matrixSumBias(Matrix m1, Matrix m2)
{
    Matrix result = createMatrix(m1->lines, m2->columns);
    int i, j;

    for (i = 0; i < m1->lines; i++)
    {
        for (j = 0; j < m2->columns; j++)
        {
            result->values[i][j] = m1->values[i][j] + m2->values[0][j];
        }
    }
    return result;
}

Matrix matrixSubtraction(Matrix m1, Matrix m2)
{
    Matrix result = createMatrix(m1->lines, m2->columns);
    int i, j;

    for (i = 0; i < m1->lines; i++)
    {
        for (j = 0; j < m2->columns; j++)
        {
            result->values[i][j] = m1->values[i][j] - m2->values[i][j];
        }
    }
    return result;
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

Matrix matrixTranspose(Matrix m)
{
    Matrix result = createMatrix(m->columns, m->lines);
    int i, j;

    for (i = 0; i < result->lines; i++)
    {
        for (j = 0; j < result->columns; j++)
        {
            result->values[i][j] = m->values[j][i];
        }
    }
    return result;
}

Matrix matrixMultiplyByEscalar(Matrix m, double x)
{
    Matrix result = createMatrix(m->lines, m->columns);
    int i, j;

    for (i = 0; i < m->lines; i++)
    {
        for (j = 0; j < m->columns; j++)
        {
            result->values[i][j] = m->values[i][j] * x;
        }
    }
    return result;
}

Matrix activate(Matrix M, int prime, char activation[4])
{
    Matrix result = createMatrix(M->lines, M->columns);
    int i, j;

    for (i = 0; i < M->lines; i++)
    {
        for (j = 0; j < M->columns; j++)
        {
            if (prime == 1)
            {
                if (strcmp(activation, "tanh")==0)
                {
                    result->values[i][j] = Tanh(M->values[i][j], 1);
                }
                else if (strcmp(activation, "sigm")==0)
                {
                    result->values[i][j] = Sigmoid(M->values[i][j], 1);
                }
                
            }
            else
            {
                if (strcmp(activation, "tanh")==0)
                {
                    result->values[i][j] = Tanh(M->values[i][j], 0);
                }
                else if (strcmp(activation, "sigm")==0)
                {
                    result->values[i][j] = Sigmoid(M->values[i][j], 0);
                }
                
            }
        }
    }
    return result;
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
