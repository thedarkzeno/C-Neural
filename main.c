#include <stdio.h>
#include "model.h"


int main()
{

    Model model = createModel();

    modelAddLayer(model, 2, 2, "tanh");

    modelAddLayer(model, 2, 2, "tanh");
    
    modelAddLayer(model, 2, 1, "tanh");

    Matrix input = createMatrix(4, 2);

    Matrix output = createMatrix(4, 1);

    // initMatrixRandom(input);
    double values1[4][2] = {{0, 0}, {1, 1},{0, 1}, {1, 0}};
    

    double out1[4][1] = {{0}, {0}, {1}, {1}};


    initMatrix(input, values1);
    initMatrix(output, out1);

    
    
    modelFit(model, input, output, .01, 150000);

    printf("\n");

    printf("Results after training\n");

    printMatrix(modelForward(model, input));

    saveModel(model);

    system("pause");

    return 0;
}