struct matrix
{
    int lines;
    int columns;
    double **values;
};

typedef struct matrix *Matrix;

Matrix createMatrix(int lines, int columns);

double getMatrixElement(Matrix M, int line, int column);

void initMatrix(Matrix M, double *values);

void initMatrixRandom(Matrix M);

Matrix matrixMultiplication(Matrix m1, Matrix m2);

Matrix matrixDotProduct(Matrix m1, Matrix m2);

Matrix matrixSum(Matrix m1, Matrix m2);

Matrix matrixSumBias(Matrix m1, Matrix m2);

Matrix matrixSubtraction(Matrix m1, Matrix m2);

Matrix matrixClone(Matrix m);

Matrix matrixTranspose(Matrix m);

Matrix matrixMultiplyByEscalar(Matrix m, double x);

Matrix activate(Matrix M, int prime, char activation[4]);

void printMatrix(Matrix M);

double MatrixSumValues(Matrix M);