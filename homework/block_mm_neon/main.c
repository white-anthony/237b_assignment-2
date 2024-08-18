#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>

#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void BlockMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result) {
    int rows_a = input0->shape[0];
    int cols_a = input0->shape[1];
    int rows_b = input1->shape[0];
    int cols_b = input1->shape[1];

    if (cols_a != rows_b) {
        printf("ERROR!\n");
        return; // Invalid matrices for multiplication
    }

    const int block_size = 4;

    result->shape[0] = rows_a;
    result->shape[1] = cols_b;
    result->data = (float*)calloc(rows_a * cols_b, sizeof(float));

    for (int i = 0; i < rows_a; i += block_size) {
        for (int j = 0; j < cols_b; j += block_size) {
            for (int k = 0; k < cols_a; k += block_size) {
                // Process each block
                for (int bi = 0; bi < block_size && (i + bi) < rows_a; ++bi) {
                    for (int bj = 0; bj < block_size && (j + bj) < cols_b; ++bj) {
                        float32x4_t c_vec = vld1q_f32(result->data + (i + bi) * cols_b + j + bj);

                        for (int bk = 0; bk < block_size && (k + bk) < cols_a; ++bk) {
                            float32x4_t a_vec = vld1q_f32(input0->data + (i + bi) * cols_a + k + bk);
                            float32x4_t b_vec = vld1q_f32(input1->data + (k + bk) * cols_b + j + bj);

                            // Compute element-wise multiplication and accumulate
                            c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                        }

                        vst1q_f32(result->data + (i + bi) * cols_b + j + bj, c_vec);
                    }
                }
            }
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

    // Initialize result matrix
    for (int i = 0; i < rows * cols; i++)
    {
        host_c.data[i] = 0.00f;
    }
    
    BlockMatrixMultiply(&host_a, &host_b, &host_c);

    // // Check the result of the matrix multiply
    // if (CompareMatrices(&answer, &host_c))
    // {
    //     printf("Matrix multiplication result is correct!\n");
    // }
    // else
    // {
    //     printf("Matrix multiplication result is incorrect.\n");
    // }

    // Check the result of the matrix multiply
    CheckMatrix(&answer, &host_c);

    // Save the matrix
    SaveMatrix(input_file_d, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}
