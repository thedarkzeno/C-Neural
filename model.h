#include "matrix.h"


struct layer
{
    Matrix weights;
    Matrix bias;
    Matrix logit; //last result previous to activation
    Matrix input;
    char *activation;
    struct layer *next; //next layer
};

typedef struct layer *Layer;

struct model
{
    struct layer *first;
    int nLayers;
};


typedef struct model *Model;

double loss(Matrix output, Matrix label);

Layer *createLayer(int input, int output);

void insertLayer(Model m, int input, int output);

Matrix layerForward(Matrix input, Layer layer);

void layerUpdate(Layer layer, Matrix diff, double lr);

Model createModel();

int modelAddLayer(Model model, int input, int output, char activation[4]);

Matrix modelForward(Model m, Matrix input);

void modelFit(Model m, Matrix input, Matrix labels, double lr, int epochs);

void saveModel(Model m);