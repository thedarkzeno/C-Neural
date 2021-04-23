#include <stdlib.h>
#include <stdio.h>
#include "model.h"

void cost(Matrix output, Matrix label, Matrix result)
{
    // Matrix result = createMatrix(label->lines, label->columns);
    matrixSubtraction(label, output, result);
    // return result;
}

double loss(Matrix output, Matrix label)
{
    Matrix sub = createMatrix(label->lines, label->columns);
    matrixSubtraction(label, output, sub);
    Matrix d = createMatrix(sub->lines, sub->columns);
    matrixDotProduct(sub, sub, d);
    matrixMultiplyByEscalar(d, 0.5, d);
    
    double sum = MatrixSumValues(d);

    freeMatrix(sub);
    freeMatrix(d);
    return sum / (label->columns * label->lines);
}

Matrix layerForward(Matrix input, Layer layer)
{
    Matrix result = createMatrix(input->lines, layer->weights->columns);
    Matrix m = createMatrix(input->lines, layer->weights->columns);
    matrixMultiplication(input, layer->weights, m);
    matrixSumBias(m, layer->bias, result);
    layer->logit = result;
    layer->input = input;
    activate(result, 0, layer->activation, result);
    freeMatrix(m);
    return result;
}

void layerGradient(Layer layer, Matrix diff, Matrix result)
{
    Matrix a = createMatrix(layer->logit->lines, layer->logit->columns);
    activate(layer->logit, 1, layer->activation, a);
    Matrix p = createMatrix(diff->lines, diff->columns);
    matrixDotProduct(diff, a, p);
    // Matrix m = createMatrix(p->lines, layer->weights->lines);
    Matrix mt = createMatrix(layer->weights->columns, layer->weights->lines);
    matrixTranspose(layer->weights, mt);
    matrixMultiplication(p, mt, result);
    freeMatrix(a);
    freeMatrix(mt);
    freeMatrix(p);
    // return m;
}

void layerUpdate(Layer layer, Matrix diff, double lr)
{

    Matrix p = createMatrix(diff->lines, diff->columns);
    Matrix pt = createMatrix(p->columns, p->lines);
    Matrix l_a = createMatrix(layer->logit->lines, layer->logit->columns);
    activate(layer->logit, 1, layer->activation, l_a);
    matrixDotProduct(diff, l_a, p);
    matrixTranspose(p, pt);

    Matrix g = createMatrix(p->columns, layer->input->columns);
    Matrix gt = createMatrix(g->columns, g->lines);
    matrixMultiplication(pt, layer->input, g);
    matrixTranspose(g, gt);
    matrixMultiplyByEscalar(gt, lr, gt);
    matrixMultiplyByEscalar(p, lr, p);

    matrixSum(layer->weights, gt, layer->weights);
    matrixSum(layer->bias, p, layer->bias);

    freeMatrix(p);
    freeMatrix(pt);
    freeMatrix(g);
    freeMatrix(gt);
    freeMatrix(l_a);
}

void modelFit(Model m, Matrix input, Matrix labels, double lr, int epochs)
{

    struct layer *layer = m->first;
    int i, j, e;
    int num_bar = epochs / 10;
    for (e = 0; e < epochs; e++)
    {
        Matrix output = modelForward(m, input);
        struct matrix *diff = createMatrix(output->lines, output->columns);
        cost(output, labels, diff);
        for (j = 0; j < m->nLayers; j++)
        {
            layer = m->first;
            for (i = 0; i < m->nLayers - j; i++)
            {

                if (i == m->nLayers - 1 - j)
                {

                    layerUpdate(layer, diff, lr);
                    Matrix res = createMatrix(diff->lines, layer->weights->lines);
                    layerGradient(layer, diff, res);
                    freeMatrix(diff);
                    diff = matrixClone(res);
                    freeMatrix(res);
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
        freeMatrix(output);
        freeMatrix(diff);
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
