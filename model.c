#include <stdlib.h>
#include <stdio.h>
#include "model.h"

Matrix cost(Matrix output, Matrix label)
{
    return matrixSubtraction(label, output);
}

double loss(Matrix output, Matrix label)
{
    Matrix sub = matrixSubtraction(label, output);
    return MatrixSumValues(matrixMultiplyByEscalar(matrixDotProduct(sub, sub), 0.5)) / (label->columns * label->lines);
}

Matrix layerForward(Matrix input, Layer layer)
{
    Matrix result = createMatrix(input->lines, layer->weights->columns);
    result = matrixSumBias(matrixMultiplication(input, layer->weights), layer->bias);
    layer->logit = result;
    layer->input = input;
    result = activate(result, 0, layer->activation);
    return result;
}

Matrix layerGradient(Layer layer, Matrix diff)
{
    Matrix a = activate(layer->logit, 1, layer->activation);
    Matrix p = matrixDotProduct(diff, a);
    return matrixMultiplication(p, matrixTranspose(layer->weights));
}

void layerUpdate(Layer layer, Matrix diff, double lr)
{

    Matrix p = matrixDotProduct(diff, activate(layer->logit, 1, layer->activation));

    Matrix g = matrixMultiplication(matrixTranspose(p), layer->input);

    layer->weights = matrixSum(layer->weights, matrixMultiplyByEscalar(matrixTranspose(g), lr));
    layer->bias = matrixSum(layer->bias, matrixMultiplyByEscalar(p, lr));
}

void modelFit(Model m, Matrix input, Matrix labels, double lr, int epochs)
{

    struct layer *layer = m->first;
    int i, j, e;
    int num_bar = epochs / 10;
    for (e = 0; e < epochs; e++)
    {
        Matrix output = modelForward(m, input);
        struct matrix *diff = cost(output, labels);
        for (j = 0; j < m->nLayers; j++)
        {
            layer = m->first;
            for (i = 0; i < m->nLayers - j; i++)
            {

                if (i == m->nLayers - 1 - j)
                {

                    layerUpdate(layer, diff, lr);
                    diff = layerGradient(layer, diff);
                }
                layer = layer->next;
            }
        }

        if (num_bar != 0 && e % num_bar == 0)
        {
            printf("progress: [");
            int p;
            for (p = 0; p < e / num_bar; p++)
            {
                printf("=");
            }
            for (p = 0; p < 9 - e / num_bar; p++)
            {
                printf(" ");
            }

            printf("]\n");
            output = modelForward(m, input);
            printf("Loss: %f  |  Epoch: %d\n", loss(output, labels), e);
        }
    }
    printf("\n");
}

Model createModel()
{
    Model new = malloc(sizeof(Model));
    if (!new)
        return NULL;
    new->nLayers = 0;
    new->first = NULL;

    return new;
}

static struct layer *next_layer(struct layer *n)
{
    return n->next;
}

static struct layer *new_layer_top(struct layer *n, int input, int output, char activation[4])
{
    struct layer *new_layer = malloc(sizeof(struct layer));
    new_layer->next = n;
    new_layer->weights = createMatrix(input, output);
    initMatrixRandom(new_layer->weights);
    new_layer->bias = createMatrix(1, output);
    initMatrixRandom(new_layer->bias);
    new_layer->logit = createMatrix(1, output);

    new_layer->activation = activation;

    return new_layer;
}

int modelAddLayer(Model m, int input, int output, char activation[4])
{

    int index = m->nLayers;
    int success = 1;
    if (!m)
        return -1;

    if (index == 0)
    {
        struct layer *n = new_layer_top(m->first, input, output, activation);
        m->nLayers++;
        m->first = n;
        return 1;
    }

    int in;
    struct layer *n1 = m->first;

    for (in = 0; in <= index - 1; in++)
    {

        if (in < index - 1)
        {
            n1 = next_layer(n1);
        }
    }

    if (success == 1)
    {

        struct layer *new = new_layer_top(n1->next, input, output, activation);
        n1->next = new;
        m->nLayers++;
    }

    return success;
}

Matrix modelForward(Model m, Matrix input)
{
    struct layer *layer = m->first;
    struct matrix *Input = input;
    int i = 0;
    while (layer != NULL)
    {
        Input = layerForward(Input, layer);
        layer = layer->next;
        i++;
    }
    return Input;
}

void saveModel(Model m)
{
    struct layer *layer = m->first;
    FILE *fp;
    fp = fopen("./model.json", "w+");
    fprintf(fp, "{ \n");
    int l = 0;
    while (layer != NULL)
    {
        fprintf(fp, "\"layer_%d\": { \n", l);

        int i, j;
        fprintf(fp, "\"input\": \"%d\",\n", layer->weights->lines);
        fprintf(fp, "\"output\": \"%d\",\n", layer->bias->columns);
        //save weights
        fprintf(fp, "\"weights\": [ ");
        for (i = 0; i < layer->weights->lines; i++)
        {
            fprintf(fp, "[ ");
            for (j = 0; j < layer->weights->columns; j++)
            {
                if (j != layer->weights->columns - 1)
                {
                    fprintf(fp, "%f, ", layer->weights->values[i][j]);
                }
                else
                {
                    fprintf(fp, "%f", layer->weights->values[i][j]);
                }
            }
            if (i != layer->weights->lines - 1)
            {
                fprintf(fp, "], ");
            }
            else
            {
                fprintf(fp, "] ");
            }
            // printf("\n");
        }
        fprintf(fp, "],\n");

        //save bias
        fprintf(fp, "\"bias\": [ ");
        for (i = 0; i < layer->bias->lines; i++)
        {
            fprintf(fp, "[ ");
            for (j = 0; j < layer->bias->columns; j++)
            {
                if (j != layer->bias->columns - 1)
                {
                    fprintf(fp, "%f, ", layer->bias->values[i][j]);
                }
                else
                {
                    fprintf(fp, "%f", layer->bias->values[i][j]);
                }
            }
            if (i != layer->bias->lines - 1)
            {
                fprintf(fp, "], ");
            }
            else
            {
                fprintf(fp, "] ");
            }
            
            // printf("\n");
        }
        fprintf(fp, "], \n");

        fprintf(fp, "\"activation\": \"%s\"", layer->activation);

        fprintf(fp, "}");
        if (layer->next != NULL)
        {
            fprintf(fp, ", \n");
        }
        else
        {
            fprintf(fp, "\n");
        }

        layer = layer->next;
        l++;
    }
    fprintf(fp, "}\n");
    fclose(fp);
}
