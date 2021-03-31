#include <math.h>
#include "activation.h"



double Sigmoid(double x, int prime)
{
     if (prime == 1)
     {
          return Sigmoid(x, 0) * (1 - Sigmoid(x, 0));
     }
     double result;
     result = 1 / (1 + exp(-x));
     return result;
}

double Tanh(double x, int prime)
{
     if (prime == 1)
     {
          return 2 * Sigmoid(x, 0) * (1 - Sigmoid(x, 0));
     }
     double result;
     result = 2 * Sigmoid(x,0) - 1;
     return result;
}

float Relu(float x, int prime)
{
     if (prime == 1)
     {
          return x;
     }
     float result;
     if(x>0) result = x;
     else result = 0;
     return result;
}