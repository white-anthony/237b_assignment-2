#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define EPSILON 1e-4  // Increased tolerance

void NaiveMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result)
{
    int rows_a = input0->shape[0];
    int cols_a = input0->shape[1];
    int rows_b = input1->shape[0];
    int cols_b = input1->shape[1];

    if (cols_a != rows_b)
    {
        fprintf(stderr, "Matrix multiplication error: columns of A (%d) != rows of B (%d)\n", cols_a, rows_b);
        exit(EXIT_FAILURE);
    }

    //printf("Matrix A dimensions: %d x %d\n", rows_a, cols_a);
    //printf("Matrix B dimensions: %d x %d\n", rows_b, cols_b);

    for (int i = 0; i < rows_a; i++)
    {
        for (int j = 0; j < cols_b; j++)
        {
            result->data[i * cols_b + j] = 0.0f;
            for (int k = 0; k < cols_a; k++)
            {
                result->data[i * cols_b + j] += input0->data[i * cols_a + k] * input1->data[k * cols_b + j];
            }
            
            //printf("result[%d][%d] = %f\n", i, j, result->data[i * cols_b + j]);
        }
    }
}

// #define EPSILON 1e-4  
// #define ABSOLUTE_EPSILON 1e-5  

// int CompareMatrices(Matrix *expected, Matrix *result)
// {
//     int size = expected->shape[0] * expected->shape[1];
//     for (int i = 0; i < size; i++)
//     {
//         float relative_error = fabs((expected->data[i] - result->data[i]) / expected->data[i]);
//         float tolerance = EPSILON * fabs(expected->data[i]);
//         float absolute_error = fabs(expected->data[i] - result->data[i]);

//         if (relative_error > tolerance && absolute_error > ABSOLUTE_EPSILON)
//         {
//             fprintf(stderr, "Mismatch at index %d: expected %f, got %f (relative error: %f, absolute error: %f)\n",
//                     i, expected->data[i], result->data[i], relative_error, absolute_error);
//             return 0; // Matrices do not match
//         }
//     }

//     return 1; // Matrices match
// }

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;

    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_c, &answer);
    CHECK_ERR(err, "LoadMatrix");

    int rows, cols;
    // Set output matrix dimensions
    rows = host_a.shape[0];
    cols = host_b.shape[1];

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (float *)malloc(sizeof(float) * host_c.shape[0] * host_c.shape[1]);

    printf("Multiplying matrices...\n");
    

    NaiveMatrixMultiply(&host_a, &host_b, &host_c);
    CheckMatrix(&answer, &host_c);

    // // Check the result of the matrix multiply
    // if (CompareMatrices(&answer, &host_c))
    // {
    //     printf("Matrix multiplication result is correct!\n");
    // }
    // else
    // {
    //     printf("Matrix multiplication result is incorrect.\n");
    // }

    // Save the matrix
    SaveMatrix(input_file_d, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}
