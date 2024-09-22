// matvect_mult.cu

#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel for matrix-vector multiplication
__global__ void matvecMulKernel(float* matrix, float* vector, float* result, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float dot_product = 0.0f;
        for (int col = 0; col < N; col++) {
            dot_product += matrix[row * N + col] * vector[col];
        }
        result[row] = dot_product;
    }
}

// Host function to call the CUDA kernel
extern "C" void matvecMul(float* matrix, float* vector, float* result, int N) {
    float *d_matrix, *d_vector, *d_result;

    // Allocate memory on the GPU
    cudaMalloc(&d_matrix, N * N * sizeof(float));
    cudaMalloc(&d_vector, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_matrix, matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with N threads, 256 per block
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    matvecMulKernel<<<numBlocks, blockSize>>>(d_matrix, d_vector, d_result, N);

    // Copy the result back to the host (CPU)
    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
}
