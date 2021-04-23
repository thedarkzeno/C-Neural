struct matrix
{
    int lines;
    int columns;
    double **values;
};

typedef struct matrix *Matrix;

Matrix createMatrix(int lines, int columns);

void freeMatrix(Matrix M);

double getMatrixElement(Matrix M, int line, int column);

void initMatrix(Matrix M, double *values);

void initMatrixRandom(Matrix M);

void matrixMultiplication(Matrix m1, Matrix m2, Matrix result);

void matrixDotProduct(Matrix m1, Matrix m2, Matrix result);

void matrixSum(Matrix m1, Matrix m2, Matrix result);

void matrixSumBias(Matrix m1, Matrix m2, Matrix result);

void matrixSubtraction(Matrix m1, Matrix m2, Matrix result);

Matrix matrixClone(Matrix m);

void matrixTranspose(Matrix m, Matrix result);

void matrixMultiplyByEscalar(Matrix m, double x, Matrix result);

void activate(Matrix M, int prime, char activation[4], Matrix result);

void printMatrix(Matrix M);

double MatrixSumValues(Matrix M);

int matrixCompareMax(Matrix m1, Matrix m2);